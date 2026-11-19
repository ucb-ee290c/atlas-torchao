package atlas.mxu

import chisel3._
import chisel3.util._
import chisel3.simulator.EphemeralSimulator._
import atlas.common._
import atlas.sa.SystolicArrayTop
import atlas.ipt.InnerProductTreesTop
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files, Paths}

// Script-driven copy of npu-model's NpuModelMatrixTraceHarness/Test: same wiring and per-cycle sampling,
// but the MxuCmd sequence and MREG contents come from a JSON file (MXU_SCRIPT) and the trace goes to MXU_OUT.
//   MXU_SCRIPT: {"sa":bool, "cycles":int,
//                "commands":[[cycle,op,bank,acc,weightSlot(,scaleE8M0)],...],
//                "mem":{"<bank>":[32 hex rows]},          initial MREG contents (missing banks are zero)
//                "loads":[[cycle,bank,[32 hex rows]],...]}  MREG bank overwritten before that cycle is sampled
//   MXU_OUT (one trace file per script): {"commands":..., "events":[{"cycle","reads":[[bank,row]],"writes":[[bank,row,hex]]}]}
// Event sampling is identical to the npu-model harness, so its fixtures are reproduced byte for byte.
class AtlasMxuTraceHarness(sa: Boolean) extends Module {
  val p = MregParams()
  val io = IO(new Bundle {
    val cmd = Flipped(Valid(new MxuCmd(p.mregIdBits)))
    val readData = Input(Vec(2, UInt(256.W)))
    val reads = Vec(2, Valid(new MregReadReq(p)))
    val writes = Vec(2, Valid(new MregWriteReq(p)))
  })
  val core = if (sa) Module(new SystolicArrayTop(SystolicArrayParams(), p)).io
             else Module(new InnerProductTreesTop(InnerProductTreeParams(), p)).io
  core.cmd <> io.cmd
  io.reads(0) := core.mregReadReq0
  io.reads(1) := core.mregReadReq1
  io.writes(0) := core.mregWriteReq0
  io.writes(1) := core.mregWriteReq1
  core.mregReadResp0.valid := RegNext(io.reads(0).valid, false.B)
  core.mregReadResp1.valid := RegNext(io.reads(1).valid, false.B)
  core.mregReadResp0.bits := RegNext(io.readData(0))
  core.mregReadResp1.bits := RegNext(io.readData(1))
}

class AtlasMxuScriptTest extends AnyFlatSpec {
  // MXU_SCRIPT / MXU_OUT: comma-separated lists of equal length. All scripts of a call share ONE Verilator build and
  // simulation instance (the SA build takes minutes); the DUT is reset (2 cycles, as in npu-model's harness) before each.
  it should "run scripted MxuCmd sequences on the actual MXU and record the MREG traces" in {
    val scripts = sys.env("MXU_SCRIPT").split(",").zip(sys.env("MXU_OUT").split(",")).toSeq.map { case (s, o) =>
      (ujson.read(Files.readString(Paths.get(s))), o) }
    val sa = scripts.head._1("sa").bool
    require(scripts.forall(_._1("sa").bool == sa), "all scripts of one run must target the same unit")
    simulate(new AtlasMxuTraceHarness(sa)) { dut =>
      for ((script, outPath) <- scripts) {
        val cycles = script("cycles").num.toInt
        val commands = script("commands").arr.map(_.arr.map(_.num.toInt).toSeq).toSeq
        val cmdAt = commands.map(c => c(0) -> c).toMap
        val loads = script.obj.get("loads").map(_.arr.toSeq).getOrElse(Seq()).map(l => (l(0).num.toInt, l(1).num.toInt, l(2).arr.map(h => BigInt(h.str, 16)).toSeq))
        val loadsAt = loads.groupBy(_._1)
        val mem = Array.fill(64, 32)(BigInt(0))
        script.obj.get("mem").foreach(_.obj.foreach { case (b, rows) => rows.arr.zipWithIndex.foreach { case (h, r) => mem(b.toInt)(r) = BigInt(h.str, 16) } })
        val events = ujson.Arr()
        dut.io.cmd.valid.poke(false.B)
        dut.io.cmd.bits.op.poke(MxuOp.PushWeight)
        dut.io.cmd.bits.mregId.poke(0.U)
        dut.io.cmd.bits.accSel.poke(false.B)
        dut.io.cmd.bits.weightSlot.poke(false.B)
        dut.io.cmd.bits.scaleE8M0.poke(127.U)
        dut.io.readData.foreach(_.poke(0.U))
        dut.reset.poke(true.B)
        dut.clock.step(2)
        dut.reset.poke(false.B)
        for (cycle <- 1 to cycles) {
          loadsAt.getOrElse(cycle, Seq()).foreach { case (_, bank, rows) => rows.zipWithIndex.foreach { case (v, r) => mem(bank)(r) = v } }
          val command = cmdAt.get(cycle)
          dut.io.cmd.valid.poke(command.nonEmpty.B)
          command.foreach { c =>
            dut.io.cmd.bits.op.poke(MxuOp.all.find(_.litValue == c(1)).get)
            dut.io.cmd.bits.mregId.poke(c(2).U)
            dut.io.cmd.bits.accSel.poke((c(3) == 1).B)
            dut.io.cmd.bits.weightSlot.poke((c(4) == 1).B)
            dut.io.cmd.bits.scaleE8M0.poke((if (c.length > 5) c(5) else 127).U)
          }
          val reads = ujson.Arr()
          val writes = ujson.Arr()
          for (port <- 0 until 2) {
            val r = dut.io.reads(port)
            if (r.valid.peek().litToBoolean) {
              val bank = r.bits.mregId.peek().litValue.toInt
              val row = r.bits.row.peek().litValue.toInt
              reads.value += ujson.Arr(bank, row)
              dut.io.readData(port).poke(mem(bank)(row).U)
            } else dut.io.readData(port).poke(0.U)
          }
          for (port <- 0 until 2) {
            val w = dut.io.writes(port)
            if (w.valid.peek().litToBoolean) {
              val bank = w.bits.mregId.peek().litValue.toInt
              val row = w.bits.row.peek().litValue.toInt
              val data = w.bits.data.peek().litValue
              writes.value += ujson.Arr(bank, row, data.toString(16))
              mem(bank)(row) = data
            }
          }
          events.value += ujson.Obj("cycle" -> cycle, "reads" -> reads, "writes" -> writes)
          dut.clock.step()
        }
        val out = ujson.Obj("commands" -> ujson.Arr.from(commands.map(c => ujson.Arr.from(c))), "events" -> events)
        Files.writeString(Paths.get(outPath), ujson.write(out, indent = 2) + "\n")
      }
    }
  }
}
