package atlas.vector

import chisel3._
import chisel3.util._
import chisel3.simulator.EphemeralSimulator._
import atlas.common._
import atlas.scalar.VpuCmd
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files, Paths}

// Script-driven VectorEngineTop run: the wiring and per-cycle sampling of npu-model's NpuModelVectorTraceHarness,
// but each run's operation, operands and MREG contents come from JSON.
//   VPU_SCRIPT: {"runs":[{"op":int,"vs1":int,"vs2":int,"vd":int,"scale":int,"imm":int,"cycles":int,
//                         "mem":{"<bank>":[32 hex rows of 256 bits]}}]}      (missing banks and rows are zero)
//   VPU_OUT:    [{"writes":[[cycle,bank,row,hex]...]}]  one entry per run (reset, one command at cycle 1, then `cycles` cycles)
class AtlasVpuTraceHarness extends Module {
  val p = MregParams()
  val io = IO(new Bundle {
    val cmd = Flipped(Valid(new VpuCmd))
    val readData = Input(Vec(2, UInt(256.W)))
    val reads = Vec(2, Valid(new MregReadReq(p)))
    val writes = Vec(2, Valid(new MregWriteReq(p)))
  })
  val vpu = Module(new VectorEngineTop(VpuParams(), p))
  vpu.io.cmd <> io.cmd
  io.reads(0) := vpu.io.mregReadReq0
  io.reads(1) := vpu.io.mregReadReq1
  io.writes(0) := vpu.io.mregWriteReq0
  io.writes(1) := vpu.io.mregWriteReq1
  vpu.io.mregReadResp0.valid := RegNext(io.reads(0).valid, false.B)
  vpu.io.mregReadResp1.valid := RegNext(io.reads(1).valid, false.B)
  vpu.io.mregReadResp0.bits := RegNext(io.readData(0))
  vpu.io.mregReadResp1.bits := RegNext(io.readData(1))
}

class AtlasVpuScriptTest extends AnyFlatSpec {
  it should "run scripted VPU operations on the actual VectorEngineTop and record the MREG writes" in {
    val runs = ujson.read(Files.readString(Paths.get(sys.env("VPU_SCRIPT"))))("runs").arr
    val results = ujson.Arr()
    simulate(new AtlasVpuTraceHarness) { dut =>
      for (run <- runs) {
        val mem = Array.fill(64, 32)(BigInt(0))
        run("mem").obj.foreach { case (b, rows) => rows.arr.zipWithIndex.foreach { case (h, r) => mem(b.toInt)(r) = BigInt(h.str, 16) } }
        dut.io.cmd.valid.poke(false.B)
        dut.io.cmd.bits.op.poke(run("op").num.toInt.U)
        dut.io.cmd.bits.vs1.poke(run("vs1").num.toInt.U)
        dut.io.cmd.bits.vs2.poke(run("vs2").num.toInt.U)
        dut.io.cmd.bits.vd.poke(run("vd").num.toInt.U)
        dut.io.cmd.bits.scaleE8M0.poke(run("scale").num.toInt.U)
        dut.io.cmd.bits.imm.poke(run("imm").num.toInt.U)
        dut.io.readData.foreach(_.poke(0.U))
        dut.reset.poke(true.B)
        dut.clock.step(2)
        dut.reset.poke(false.B)
        val writes = ujson.Arr()
        for (cycle <- 1 to run("cycles").num.toInt) {
          dut.io.cmd.valid.poke((cycle == 1).B)
          for (port <- 0 until 2) {
            val read = dut.io.reads(port)
            if (read.valid.peek().litToBoolean) {
              val bank = read.bits.mregId.peek().litValue.toInt
              val row = read.bits.row.peek().litValue.toInt
              dut.io.readData(port).poke(mem(bank)(row).U)
            } else dut.io.readData(port).poke(0.U)
          }
          for (port <- 0 until 2) {
            val w = dut.io.writes(port)
            if (w.valid.peek().litToBoolean) {
              val bank = w.bits.mregId.peek().litValue.toInt
              val row = w.bits.row.peek().litValue.toInt
              val data = w.bits.data.peek().litValue
              writes.value += ujson.Arr(cycle, bank, row, data.toString(16))
              mem(bank)(row) = data
            }
          }
          dut.clock.step()
        }
        results.value += ujson.Obj("writes" -> writes)
      }
    }
    Files.writeString(Paths.get(sys.env("VPU_OUT")), ujson.write(results) + "\n")
  }
}
