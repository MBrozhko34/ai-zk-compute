pragma circom 2.1.6;

include "circomlib/circuits/comparators.circom";

template Boolify() {
    signal input  in;
    signal output out;
    out <== in;
    // in ∈ {0,1}
    in * (in - 1) === 0;
}

template RangeLtConst(nBits, CONST) {
    // Enforce 0 <= in < CONST where CONST < 2^nBits
    signal input  in;
    component lt = LessThan(nBits);
    lt.in[0] <== in;
    lt.in[1] <== CONST;
    signal output ok;
    ok <== lt.out;   // == 1
    ok === 1;
}

template StepNonNeg(nBits) {
    // out = 1 if pos >= neg else 0 (unsigned compare)
    signal input  pos;
    signal input  neg;
    signal output out;
    component ge = GreaterEqThan(nBits);
    ge.in[0] <== pos;
    ge.in[1] <== neg;
    out      <== ge.out;
}

// 1-sample forward
//
// Flattened weight layout (17 params):
// W1: [0..7]  (j*2 + {0,1})
// B1: [8..11]
//  V: [12..15]
// b2: [16]
//
template MlpForwardUnsigned() {
    signal input w_abs[17];   // magnitudes 0..127
    signal input w_sign[17];  // sign bits {0,1} ; 1 = negative
    signal input x0;          // feature 0..15
    signal input x1;          // feature 0..15
    signal output pred;       // 0/1

    // ranges
    component absOK[17];
    component sOK  [17];
    for (var i=0; i<17; i++) {
        absOK[i] = RangeLtConst(8, 128); absOK[i].in <== w_abs[i];
        sOK[i]   = Boolify();            sOK[i].in   <== w_sign[i];
    }
    component x0OK = RangeLtConst(4, 16); x0OK.in <== x0;
    component x1OK = RangeLtConst(4, 16); x1OK.in <== x1;

    // precompute (1 - sign) for gating
    signal oneMinusSign[17];
    for (var i=0; i<17; i++) {
        oneMinusSign[i] <== 1 - w_sign[i];
    }

    // hidden gates
    signal s[4];

    // per-unit helper arrays (declared once; wired in loop)
    signal posSum[4];
    signal negSum[4];

    signal m00[4];  // |w00| * x0  (binary product)
    signal m01[4];  // |w01| * x1

    signal t00p[4]; signal t00n[4];
    signal t01p[4]; signal t01n[4];
    signal tbp [4]; signal tbn [4];

    component gate[4];

    for (var j=0; j<4; j++) {
        var base = j*2;

        // magnitude * input (binary)
        m00[j] <== w_abs[base+0] * x0;
        m01[j] <== w_abs[base+1] * x1;

        // gate by sign (binary)
        t00p[j] <== m00[j] * oneMinusSign[base+0];
        t00n[j] <== m00[j] * w_sign[base+0];

        t01p[j] <== m01[j] * oneMinusSign[base+1];
        t01n[j] <== m01[j] * w_sign[base+1];

        // bias split by sign (binary)
        tbp[j]  <== w_abs[8+j] * oneMinusSign[8+j];
        tbn[j]  <== w_abs[8+j] * w_sign[8+j];

        posSum[j] <== t00p[j] + t01p[j] + tbp[j];
        negSum[j] <== t00n[j] + t01n[j] + tbn[j];

        gate[j] = StepNonNeg(16);      // bound ~ 3937
        gate[j].pos <== posSum[j];
        gate[j].neg <== negSum[j];
        s[j]        <== gate[j].out;   // 0/1
    }

    // output layer: z = b2 + Σ vj*s[j], threshold via pos>=neg
    signal posO0; signal negO0;
    posO0 <== w_abs[16] * oneMinusSign[16];
    negO0 <== w_abs[16] * w_sign[16];

    signal mV[4];    // |vj| * s[j]  (binary; s[j]∈{0,1})
    signal vpos[4];  // + part
    signal vneg[4];  // - part
    for (var j=0; j<4; j++) {
        mV[j]   <== w_abs[12+j] * s[j];
        vpos[j] <== mV[j] * oneMinusSign[12+j];
        vneg[j] <== mV[j] * w_sign[12+j];
    }

    // prefix sums (avoid reassigning signals)
    signal posStep[5];
    signal negStep[5];
    posStep[0] <== posO0;
    negStep[0] <== negO0;
    for (var j=0; j<4; j++) {
        posStep[j+1] <== posStep[j] + vpos[j];
        negStep[j+1] <== negStep[j] + vneg[j];
    }

    component outStep = StepNonNeg(10); // bound ~ 635
    outStep.pos <== posStep[4];
    outStep.neg <== negStep[4];
    pred        <== outStep.out;
}

// batched accuracy over 256 rows

// Accuracy over 256 rows, with *private* arrays (mask/x0/x1/y) and
// w_abs/w_sign + acc_bps as inputs. This keeps your original logic.
template MlpAcc256() {
    // Public-ish inputs to the core, but we won’t mark them public here.
    signal input acc_bps;
    signal input w_abs[17];
    signal input w_sign[17];
    signal input mask[256];  // 0/1
    signal input x0  [256];  // 0..15
    signal input x1  [256];  // 0..15
    signal input y   [256];  // 0/1

    // range checks (declare arrays, then wire in loop)
    component mB  [256];
    component yB  [256];
    component x0OK[256];
    component x1OK[256];
    for (var i=0; i<256; i++) { mB[i]   = Boolify();           mB[i].in   <== mask[i]; }
    for (var i=0; i<256; i++) { yB[i]   = Boolify();           yB[i].in   <== y[i];    }
    for (var i=0; i<256; i++) { x0OK[i] = RangeLtConst(4, 16); x0OK[i].in <== x0[i];   }
    for (var i=0; i<256; i++) { x1OK[i] = RangeLtConst(4, 16); x1OK[i].in <== x1[i];   }

    // 256 forward passes (declare array; instantiate in loop)
    component f[256];
    for (var i=0; i<256; i++) {
        f[i] = MlpForwardUnsigned();
        for (var k=0; k<17; k++) {
            f[i].w_abs[k]  <== w_abs[k];
            f[i].w_sign[k] <== w_sign[k];
        }
        f[i].x0 <== x0[i];
        f[i].x1 <== x1[i];
    }

    // correctness bit per row
    signal corr[256];
    for (var i=0; i<256; i++) {
        // corr[i] = (pred == y[i]) ? 1 : 0
        corr[i] <== 1 - (f[i].pred - y[i]) * (f[i].pred - y[i]);
    }

    // prefix sums (no reassignments)
    signal activeStep[257];
    signal corrStep  [257];
    activeStep[0] <== 0;
    corrStep[0]   <== 0;
    for (var i=0; i<256; i++) {
        activeStep[i+1] <== activeStep[i] + mask[i];
        corrStep[i+1]   <== corrStep[i]   + corr[i] * mask[i];
    }

    signal active;       active       <== activeStep[256];
    signal totalCorrect; totalCorrect <== corrStep[256];

    // active >= 1
    component ge1 = GreaterEqThan(9);
    ge1.in[0] <== active; ge1.in[1] <== 1;
    ge1.out === 1;

    // acc_bps = floor(10000 * totalCorrect / active)
    signal num;  num  <== totalCorrect * 10000;
    signal prod; prod <== acc_bps * active;

    component ge2 = GreaterEqThan(24); // num up to 2,560,000
    ge2.in[0] <== num; ge2.in[1] <== prod;
    ge2.out === 1;

    signal rem; rem <== num - prod;
    component lt2 = LessThan(9); // active <= 256
    lt2.in[0] <== rem;
    lt2.in[1] <== active;
    lt2.out === 1;

    // passthrough keeps acc_bps as a constrained signal
    acc_bps * 1 === acc_bps;
}


// Compute packed representations (32 entries per limb).
// These will be *public inputs* in the main wrapper.
template PackBits256() {
    signal input  b[256];     // each {0,1}
    signal output limbs[8];   // 8 limbs, 32 bits each

    // range checks
    component bOK[256];
    for (var i=0; i<256; i++) { bOK[i] = Boolify(); bOK[i].in <== b[i]; }

    // prefix sums per limb (no re-declarations inside loops)
    signal step[8][33];
    for (var g=0; g<8; g++) {
        step[g][0] <== 0;
        var base = g*32;
        var pow = 1;                 // compile-time scalar, safe to mutate
        for (var k=0; k<32; k++) {
            // step[g][k+1] = step[g][k] + b[base+k] * (2^k)
            step[g][k+1] <== step[g][k] + b[base+k] * pow;
            pow = pow * 2;
        }
        limbs[g] <== step[g][32];
    }
}

template PackU4x256() {
    signal input  v[256];     // each 0..15
    signal output limbs[8];   // 8 limbs, 32 nibbles each

    // range checks
    component rOK[256];
    for (var i=0; i<256; i++) { rOK[i] = RangeLtConst(4, 16); rOK[i].in <== v[i]; }

    // prefix sums per limb
    signal step[8][33];
    for (var g=0; g<8; g++) {
        step[g][0] <== 0;
        var base = g*32;
        var pow = 1;                 // 16^k shift (4 bits)
        for (var k=0; k<32; k++) {
            step[g][k+1] <== step[g][k] + v[base+k] * pow;
            pow = pow * 16;
        }
        limbs[g] <== step[g][32];
    }
}

// Wrapper that takes *public packed inputs* AND *private full arrays*,
// computes the packings from the private arrays, and enforces equality
// with the public limbs. Feeds everything to the core MlpAcc256.
template MlpAcc256_PackedInputs() {
    // Public inputs for verifier size:
    signal input acc_bps;
    signal input w_abs[17];
    signal input w_sign[17];
    signal input mask_p[8];   // packed mask (bits)
    signal input x0_p[8];     // packed x0 (nibbles)
    signal input x1_p[8];     // packed x1 (nibbles)
    signal input y_p[8];      // packed y (bits)

    // Private (witness) arrays:
    signal input mask[256];
    signal input x0  [256];
    signal input x1  [256];
    signal input y   [256];

    // Core
    component F = MlpAcc256();
    F.acc_bps <== acc_bps;
    for (var t=0; t<17; t++) { F.w_abs[t]  <== w_abs[t];  F.w_sign[t] <== w_sign[t]; }
    for (var i=0; i<256; i++) {
        F.mask[i] <== mask[i]; F.x0[i] <== x0[i];
        F.x1[i]   <== x1[i];   F.y[i]  <== y[i];
    }

    // Pack computed (private) arrays and constrain to public limbs
    component Pmask = PackBits256();
    component Py    = PackBits256();
    component Px0   = PackU4x256();
    component Px1   = PackU4x256();
    for (var i2=0; i2<256; i2++) {
        Pmask.b[i2] <== mask[i2];
        Py.b[i2]    <== y[i2];
        Px0.v[i2]   <== x0[i2];
        Px1.v[i2]   <== x1[i2];
    }

    for (var g=0; g<8; g++) {
        mask_p[g] === Pmask.limbs[g];
        x0_p[g]   === Px0.limbs[g];
        x1_p[g]   === Px1.limbs[g];
        y_p[g]    === Py.limbs[g];
    }
}

component main {
    public [ acc_bps, w_abs, w_sign, mask_p, x0_p, x1_p, y_p ]
} = MlpAcc256_PackedInputs();
