// scripts/plonk_calldata.js
// Usage: node scripts/plonk_calldata.js <proof.json> <public.json>
// Prints either the classic "0x...,[...]" string OR a JSON array [ proofHexOrArray, pubsArray ].
// We'll parse whatever it prints on the Python side.
const fs = require("fs");
const path = require("path");
(async () => {
  try {
    const [,, proofPath, publicPath] = process.argv;
    if (!proofPath || !publicPath) {
      console.error("Usage: node scripts/plonk_calldata.js <proof.json> <public.json>");
      process.exit(2);
    }
    const { plonk } = require("snarkjs");
    const proof = JSON.parse(fs.readFileSync(path.resolve(proofPath), "utf8"));
    const pubs  = JSON.parse(fs.readFileSync(path.resolve(publicPath), "utf8"));

    const cd = await plonk.exportSolidityCallData(proof, pubs);
    // Some versions return a string "0x...,[pubs]"
    // Some return an array [proofHexOrArray, pubs]
    if (Array.isArray(cd)) {
      console.log(JSON.stringify(cd));
    } else {
      console.log(String(cd));
    }
  } catch (e) {
    console.error(e && e.stack ? e.stack : String(e));
    process.exit(99);
  }
})();
