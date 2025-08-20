// circuits/XorCircuit.circom
pragma circom 2.1.6;
include "circomlib/circuits/comparators.circom";

/* 1-bit step: out = 1 iff in >= 1 (2-bit comparator: values 0..3) */
template Step() {
    signal input  in;
    signal output out;

    component ge = GreaterEqThan(2);
    ge.in[0] <== in;
    ge.in[1] <== 1;
    out       <== ge.out;
}

/* 2-2-1 XOR network (integer, binarised)
   public : lr, steps, acc_bps (x10_000)
   private: wIH[4], bH[2], bO (all 0/1/2; no negatives) */
template XorNet() {
    // public
    signal input lr;
    signal input steps;
    signal input acc_bps;

    // private
    signal input wIH[4];   // row-major 2x2
    signal input bH[2];
    signal input bO;       // 0/1/2

    // constants for XOR dataset
    var X0_0 = 0; var X0_1 = 0; var Y0 = 0;
    var X1_0 = 1; var X1_1 = 0; var Y1 = 1;
    var X2_0 = 0; var X2_1 = 1; var Y2 = 1;
    var X3_0 = 1; var X3_1 = 1; var Y3 = 0;

    // components/signals declared OUTSIDE any loop
    component h0s[4];
    component h1s[4];
    component outs[4];
    signal diff0; signal sq0; signal c0;
    signal diff1; signal sq1; signal c1;
    signal diff2; signal sq2; signal c2;
    signal diff3; signal sq3; signal c3;

    // sample 0
    var h00 = wIH[0]*X0_0 + wIH[1]*X0_1 + bH[0];
    var h10 = wIH[2]*X0_0 + wIH[3]*X0_1 + bH[1];
    h0s[0] = Step(); h0s[0].in <== h00;
    h1s[0] = Step(); h1s[0].in <== h10;
    var o0 = h0s[0].out - h1s[0].out + bO;
    outs[0] = Step(); outs[0].in <== o0;
    diff0 <== outs[0].out - Y0;
    sq0   <== diff0 * diff0;
    c0    <== 1 - sq0;

    // sample 1
    var h01 = wIH[0]*X1_0 + wIH[1]*X1_1 + bH[0];
    var h11 = wIH[2]*X1_0 + wIH[3]*X1_1 + bH[1];
    h0s[1] = Step(); h0s[1].in <== h01;
    h1s[1] = Step(); h1s[1].in <== h11;
    var o1 = h0s[1].out - h1s[1].out + bO;
    outs[1] = Step(); outs[1].in <== o1;
    diff1 <== outs[1].out - Y1;
    sq1   <== diff1 * diff1;
    c1    <== 1 - sq1;

    // sample 2
    var h02 = wIH[0]*X2_0 + wIH[1]*X2_1 + bH[0];
    var h12 = wIH[2]*X2_0 + wIH[3]*X2_1 + bH[1];
    h0s[2] = Step(); h0s[2].in <== h02;
    h1s[2] = Step(); h1s[2].in <== h12;
    var o2 = h0s[2].out - h1s[2].out + bO;
    outs[2] = Step(); outs[2].in <== o2;
    diff2 <== outs[2].out - Y2;
    sq2   <== diff2 * diff2;
    c2    <== 1 - sq2;

    // sample 3
    var h03 = wIH[0]*X3_0 + wIH[1]*X3_1 + bH[0];
    var h13 = wIH[2]*X3_0 + wIH[3]*X3_1 + bH[1];
    h0s[3] = Step(); h0s[3].in <== h03;
    h1s[3] = Step(); h1s[3].in <== h13;
    var o3 = h0s[3].out - h1s[3].out + bO;
    outs[3] = Step(); outs[3].in <== o3;
    diff3 <== outs[3].out - Y3;
    sq3   <== diff3 * diff3;
    c3    <== 1 - sq3;

    // total correct
    signal total;
    total <== c0 + c1 + c2 + c3;

    // claimed accuracy must match
    signal achieved;
    achieved <== total * 2500;   // 4 samples → 25% each
    achieved === acc_bps;

    // force lr & steps into pubSignals
    lr    * 1 === lr;
    steps * 1 === steps;
}

/* Inline public declaration avoids the separate 'public [...]' line */
component main { public [lr, steps, acc_bps] } = XorNet();
