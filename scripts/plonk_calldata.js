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
