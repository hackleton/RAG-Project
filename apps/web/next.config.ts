import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // `standalone` keeps the production image to the server plus the traced
  // dependencies, instead of shipping the whole node_modules tree.
  output: "standalone",
};

export default config;
