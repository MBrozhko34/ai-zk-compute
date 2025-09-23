// hardhat.config.ts
import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";
import * as dotenv from "dotenv";
dotenv.config();

const config: HardhatUserConfig = {
  solidity: {
    compilers: [
      {
        version: "0.8.28",
        settings: {
          optimizer: { enabled: true, runs: 200 },
          viaIR: true, // prevents classic "stack too deep"
        },
      },
    ],
  },
  networks: {
    hardhat: {},
    localhost: { url: process.env.RPC_URL || "http://127.0.0.1:8545" },
  },
};

export default config;
