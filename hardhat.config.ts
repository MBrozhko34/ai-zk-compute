import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";

const config: HardhatUserConfig = {
  solidity: {
    version: "0.8.28",
    settings: {
      optimizer: { enabled: true, runs: 200 },
      viaIR: true,                 // 👈 fixes “stack too deep”
      // evmVersion: "paris",      // optional, but fine to leave default
    },
  },
};

export default config;
