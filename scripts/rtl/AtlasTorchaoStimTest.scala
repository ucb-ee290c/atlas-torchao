package atlas.mxu

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import atlas.common.InnerProductTreeParams
import atlas.ipt.AnchorAccumulationTree
import sp26FPUnits.E4M3FMA
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files,Paths}

// Drives npu-model's NpuModelArithmeticHarness datapaths (E4M3FMA, AnchorAccumulationTree and the FP8
// converters) with stimuli from atlas-torchao instead of seeded random cases, under Verilator.
// L1_STIM: JSON {"cases":[{"a":[32],"w":[32],"partial":int,"scale":int}, ...]}; L1_OUT: results JSON.
class AtlasTorchaoStimHarness extends Module {
  val io = IO(new Bundle {
    val a = Input(Vec(32,UInt(8.W)))
    val w = Input(Vec(32,UInt(8.W)))
    val partial = Input(UInt(16.W))
    val scale = Input(UInt(8.W))
    val fma = Output(UInt(16.W))
    val ipt = Output(UInt(16.W))
    val quant = Output(UInt(8.W))
    val dequant = Output(UInt(16.W))
  })
  val fma = Module(new E4M3FMA)
  fma.io.a := io.a(0)
  fma.io.b := io.w(0)
  fma.io.addend16 := io.partial
  io.fma := fma.io.out16
  val ipt = Module(new AnchorAccumulationTree(InnerProductTreeParams()))
  ipt.io.act := io.a
  ipt.io.weightBuf0 := io.w
  ipt.io.weightBuf1 := io.w
  ipt.io.bufReadSel := false.B
  ipt.io.psum := io.partial
  ipt.io.accumulate := true.B
  io.ipt := ipt.io.out
  io.quant := FPUtils.bf16ToE4m3(io.partial,io.scale)
  io.dequant := FPUtils.e4m3ToBf16(io.a(0))
}

class AtlasTorchaoStimTest extends AnyFlatSpec {
  it should "run atlas-torchao stimuli through the arithmetic datapaths" in {
    val stim = ujson.read(Files.readString(Paths.get(sys.env("L1_STIM"))))("cases").arr
    val results = ujson.Arr()
    simulate(new AtlasTorchaoStimHarness) { dut =>
      dut.io.a.foreach(_.poke(0.U))
      dut.io.w.foreach(_.poke(0.U))
      dut.io.partial.poke(0.U)
      dut.io.scale.poke(127.U)
      dut.reset.poke(true.B)
      dut.clock.step(2)
      dut.reset.poke(false.B)
      for (c <- stim) {
        c("a").arr.zipWithIndex.foreach { case(v,l) => dut.io.a(l).poke(v.num.toInt.U) }
        c("w").arr.zipWithIndex.foreach { case(v,l) => dut.io.w(l).poke(v.num.toInt.U) }
        dut.io.partial.poke(c("partial").num.toInt.U)
        dut.io.scale.poke(c("scale").num.toInt.U)
        dut.clock.step(2)
        results.value += ujson.Obj("fma"->dut.io.fma.peek().litValue.toInt,"ipt"->dut.io.ipt.peek().litValue.toInt,
          "quant"->dut.io.quant.peek().litValue.toInt,"dequant"->dut.io.dequant.peek().litValue.toInt)
      }
    }
    Files.writeString(Paths.get(sys.env("L1_OUT")),ujson.write(results)+"\n")
  }
}
