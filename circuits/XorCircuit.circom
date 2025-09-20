// circuits/XorCircuit.circom
// Proves: deterministic training (quantized updates) of a fixed 2-2-1 network
// for `steps` epochs and learning-rate bucket from `lr`, then computes the
// training-set accuracy and exposes it as acc_bps.
//
// Public:  lr (ppm), steps (epochs), acc_bps (basis points)
// Private: none (weights/biases are *derived in-circuit* from fixed init + updates)

pragma circom 2.1.6;
include "circomlib/circuits/comparators.circom";

/* 1-bit step: out = 1 iff in >= 1
   Use 4-bit comparator (0..15) because hidden pre-activations can reach up to 9. */
template Step() {
    signal input  in;
    signal output out;

    component ge = GreaterEqThan(4);  // was 2 → 4 to avoid Num2Bits assert
    ge.in[0] <== in;
    ge.in[1] <== 1;
    out       <== ge.out;
}

/* Saturating updater in [0,3] with floor=0 */
template UpdateUnit() {
    signal input  x;      // 0..3
    signal input  inc;    // 0/1
    signal input  dec;    // 0/1
    signal input  delta;  // 0..3

    signal addCand;
    addCand <== x + inc * delta;    // 0..6

    // Cap at 3
    component ge3 = GreaterEqThan(3);
    ge3.in[0] <== addCand; ge3.in[1] <== 3;
    signal xCap;
    xCap <== addCand - ge3.out * (addCand - 3);

    // Subtract (saturate at 0)
    signal subAmt; subAmt <== dec * delta;
    component lt = LessThan(2);
    lt.in[0] <== xCap; lt.in[1] <== subAmt;
    signal under; under <== lt.out;

    signal xDec; xDec <== xCap - subAmt;
    signal y;
    y <== (1 - under) * xDec; // if under=1 => 0 else xDec

    signal output out;
    out <== y;
}

/* Same but saturating in [1,3] (for output bias so Step input never < 0) */
template UpdateUnitFloor1() {
    signal input  x;      // 1..3
    signal input  inc;    // 0/1
    signal input  dec;    // 0/1
    signal input  delta;  // 0..3

    signal addCand; addCand <== x + inc * delta;

    component ge3 = GreaterEqThan(3);
    ge3.in[0] <== addCand; ge3.in[1] <== 3;
    signal xCap; xCap <== addCand - ge3.out * (addCand - 3);

    signal subAmt; subAmt <== dec * delta;
    // Underflow below floor 1?  (xCap - 1) < subAmt
    component lt = LessThan(2);
    signal xCapMinus1; xCapMinus1 <== xCap - 1;
    lt.in[0] <== xCapMinus1; lt.in[1] <== subAmt;
    signal under; under <== lt.out;

    signal xDec; xDec <== xCap - subAmt;
    signal y;
    y <== (1 - under) * xDec + under * 1;

    signal output out;
    out <== y;
}

/* Training + accuracy. MAX_EPOCHS must cover the largest 'steps' in your grid. */
template XorTrain(MAX_EPOCHS) {
    // public
    signal input lr;       // ppm
    signal input steps;    // epochs (<= MAX_EPOCHS)
    signal input acc_bps;  // basis points

    // Map lr (ppm) into integer bucket in {1,2,3}
    component ge50k  = GreaterEqThan(20); ge50k.in[0] <== lr; ge50k.in[1] <== 50000;
    component ge100k = GreaterEqThan(20); ge100k.in[0] <== lr; ge100k.in[1] <== 100000;
    signal lrUnit; lrUnit <== 1 + ge50k.out + ge100k.out; // 1..3

    // XOR dataset constants (shared)
    var X0_0 = 0; var X0_1 = 0; var Y0 = 0;
    var X1_0 = 1; var X1_1 = 0; var Y1 = 1;
    var X2_0 = 0; var X2_1 = 1; var Y2 = 1;
    var X3_0 = 1; var X3_1 = 1; var Y3 = 0;

    // Epoch states (shared model)
    signal w0e[MAX_EPOCHS+1];
    signal w1e[MAX_EPOCHS+1];
    signal w2e[MAX_EPOCHS+1];
    signal w3e[MAX_EPOCHS+1];
    signal b0e[MAX_EPOCHS+1];
    signal b1e[MAX_EPOCHS+1];
    signal bOe[MAX_EPOCHS+1];

    // Initialisation
    w0e[0] <== 1; w1e[0] <== 1; w2e[0] <== 1; w3e[0] <== 1;
    b0e[0] <== 0; b1e[0] <== 0; bOe[0] <== 1;

    // Per-epoch helpers (predeclared to avoid T2011)
    component ltE[MAX_EPOCHS];
    signal   active[MAX_EPOCHS];

    // Per-epoch, per-sample rolling states
    signal sw0[MAX_EPOCHS][5];
    signal sw1[MAX_EPOCHS][5];
    signal sw2[MAX_EPOCHS][5];
    signal sw3[MAX_EPOCHS][5];
    signal sb0[MAX_EPOCHS][5];
    signal sb1[MAX_EPOCHS][5];
    signal sbO[MAX_EPOCHS][5];

    // Step components for forward passes during training
    component st0[MAX_EPOCHS][4];
    component st1[MAX_EPOCHS][4];
    component stOut[MAX_EPOCHS][4];

    // Training per-sample signals (predeclared)
    signal o   [MAX_EPOCHS][4];
    signal pos [MAX_EPOCHS][4];
    signal neg [MAX_EPOCHS][4];
    signal inc [MAX_EPOCHS][4];
    signal dec [MAX_EPOCHS][4];
    signal d0  [MAX_EPOCHS][4];
    signal d1  [MAX_EPOCHS][4];

    // Update units (predeclared)
    component uw0[MAX_EPOCHS][4];
    component uw1[MAX_EPOCHS][4];
    component uw2[MAX_EPOCHS][4];
    component uw3[MAX_EPOCHS][4];
    component ub0[MAX_EPOCHS][4];
    component ub1[MAX_EPOCHS][4];
    component ubO[MAX_EPOCHS][4]; // floor=1

    // Epoch loop
    for (var e = 0; e < MAX_EPOCHS; e++) {
        // active = (e < steps)
        ltE[e] = LessThan(10);
        ltE[e].in[0] <== e;
        ltE[e].in[1] <== steps;
        active[e] <== ltE[e].out;

        // Seed rolling states with epoch state
        sw0[e][0] <== w0e[e]; sw1[e][0] <== w1e[e]; sw2[e][0] <== w2e[e]; sw3[e][0] <== w3e[e];
        sb0[e][0] <== b0e[e]; sb1[e][0] <== b1e[e]; sbO[e][0] <== bOe[e];

        // Four samples in fixed order
        for (var s = 0; s < 4; s++) {
            // Pick dataset constants
            var x0 = (s==0?X0_0:(s==1?X1_0:(s==2?X2_0:X3_0)));
            var x1 = (s==0?X0_1:(s==1?X1_1:(s==2?X2_1:X3_1)));
            var y  = (s==0?Y0   :(s==1?Y1   :(s==2?Y2   :Y3   )));

            // Forward
            var h0 = sw0[e][s]*x0 + sw1[e][s]*x1 + sb0[e][s];
            var h1 = sw2[e][s]*x0 + sw3[e][s]*x1 + sb1[e][s];
            st0[e][s] = Step(); st0[e][s].in <== h0;
            st1[e][s] = Step(); st1[e][s].in <== h1;
            var oLin = st0[e][s].out - st1[e][s].out + sbO[e][s];
            stOut[e][s] = Step(); stOut[e][s].in <== oLin;
            o[e][s] <== stOut[e][s].out;

            // Update directions
            pos[e][s] <== (1 - o[e][s]) * y;
            neg[e][s] <== (1 - y) * o[e][s];
            inc[e][s] <== active[e] * pos[e][s];
            dec[e][s] <== active[e] * neg[e][s];

            // Deltas
            d0[e][s] <== lrUnit * x0;
            d1[e][s] <== lrUnit * x1;

            // Hidden0 follows pos/neg
            uw0[e][s] = UpdateUnit(); uw0[e][s].x <== sw0[e][s]; uw0[e][s].inc <== inc[e][s]; uw0[e][s].dec <== dec[e][s]; uw0[e][s].delta <== d0[e][s];
            uw1[e][s] = UpdateUnit(); uw1[e][s].x <== sw1[e][s]; uw1[e][s].inc <== inc[e][s]; uw1[e][s].dec <== dec[e][s]; uw1[e][s].delta <== d1[e][s];
            ub0[e][s] = UpdateUnit(); ub0[e][s].x <== sb0[e][s]; ub0[e][s].inc <== inc[e][s]; ub0[e][s].dec <== dec[e][s]; ub0[e][s].delta <== lrUnit;

            // Hidden1 opposite (because output uses s0 - s1)
            uw2[e][s] = UpdateUnit(); uw2[e][s].x <== sw2[e][s]; uw2[e][s].inc <== dec[e][s]; uw2[e][s].dec <== inc[e][s]; uw2[e][s].delta <== d0[e][s];
            uw3[e][s] = UpdateUnit(); uw3[e][s].x <== sw3[e][s]; uw3[e][s].inc <== dec[e][s]; uw3[e][s].dec <== inc[e][s]; uw3[e][s].delta <== d1[e][s];
            ub1[e][s] = UpdateUnit(); ub1[e][s].x <== sb1[e][s]; ub1[e][s].inc <== dec[e][s]; ub1[e][s].dec <== inc[e][s]; ub1[e][s].delta <== lrUnit;

            // Output bias follows hidden0 and has floor 1
            ubO[e][s] = UpdateUnitFloor1(); ubO[e][s].x <== sbO[e][s]; ubO[e][s].inc <== inc[e][s]; ubO[e][s].dec <== dec[e][s]; ubO[e][s].delta <== lrUnit;

            // Latch to next sample
            sw0[e][s+1] <== uw0[e][s].out; sw1[e][s+1] <== uw1[e][s].out; sw2[e][s+1] <== uw2[e][s].out; sw3[e][s+1] <== uw3[e][s].out;
            sb0[e][s+1] <== ub0[e][s].out; sb1[e][s+1] <== ub1[e][s].out; sbO[e][s+1] <== ubO[e][s].out;
        }

        // Commit epoch+1 state
        w0e[e+1] <== sw0[e][4]; w1e[e+1] <== sw1[e][4]; w2e[e+1] <== sw2[e][4]; w3e[e+1] <== sw3[e][4];
        b0e[e+1] <== sb0[e][4]; b1e[e+1] <== sb1[e][4]; bOe[e+1] <== sbO[e][4];
    }

    // Final (epochs beyond 'steps' are inert since active=0)
    signal fw0; fw0 <== w0e[MAX_EPOCHS];
    signal fw1; fw1 <== w1e[MAX_EPOCHS];
    signal fw2; fw2 <== w2e[MAX_EPOCHS];
    signal fw3; fw3 <== w3e[MAX_EPOCHS];
    signal fb0; fb0 <== b0e[MAX_EPOCHS];
    signal fb1; fb1 <== b1e[MAX_EPOCHS];
    signal fbO; fbO <== bOe[MAX_EPOCHS];

    // Accuracy on training set (declare components in initial scope)
    component h0s[4]; component h1s[4]; component outs[4];
    signal c0; signal c1; signal c2; signal c3;

    var H00 = fw0*X0_0 + fw1*X0_1 + fb0; h0s[0] = Step(); h0s[0].in <== H00;
    var H10 = fw2*X0_0 + fw3*X0_1 + fb1; h1s[0] = Step(); h1s[0].in <== H10;
    var O0  = h0s[0].out - h1s[0].out + fbO; outs[0] = Step(); outs[0].in <== O0;
    c0 <== 1 - (outs[0].out - Y0) * (outs[0].out - Y0);

    var H01 = fw0*X1_0 + fw1*X1_1 + fb0; h0s[1] = Step(); h0s[1].in <== H01;
    var H11 = fw2*X1_0 + fw3*X1_1 + fb1; h1s[1] = Step(); h1s[1].in <== H11;
    var O1  = h0s[1].out - h1s[1].out + fbO; outs[1] = Step(); outs[1].in <== O1;
    c1 <== 1 - (outs[1].out - Y1) * (outs[1].out - Y1);

    var H02 = fw0*X2_0 + fw1*X2_1 + fb0; h0s[2] = Step(); h0s[2].in <== H02;
    var H12 = fw2*X2_0 + fw3*X2_1 + fb1; h1s[2] = Step(); h1s[2].in <== H12;
    var O2  = h0s[2].out - h1s[2].out + fbO; outs[2] = Step(); outs[2].in <== O2;
    c2 <== 1 - (outs[2].out - Y2) * (outs[2].out - Y2);

    var H03 = fw0*X3_0 + fw1*X3_1 + fb0; h0s[3] = Step(); h0s[3].in <== H03;
    var H13 = fw2*X3_0 + fw3*X3_1 + fb1; h1s[3] = Step(); h1s[3].in <== H13;
    var O3  = h0s[3].out - h1s[3].out + fbO; outs[3] = Step(); outs[3].in <== O3;
    c3 <== 1 - (outs[3].out - Y3) * (outs[3].out - Y3);

    signal total; total <== c0 + c1 + c2 + c3;
    signal achieved; achieved <== total * 2500; // 4 samples → 25% each
    achieved === acc_bps;

    // Force lr & steps into public signals
    lr    * 1 === lr;
    steps * 1 === steps;
}

/* MAX_EPOCHS must cover your largest 'steps' in the grid (e.g., 300) */
component main { public [lr, steps, acc_bps] } = XorTrain(300);
