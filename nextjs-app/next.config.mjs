/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  // Temporarily set to false for development to fix WebSocket issues
  reactStrictMode: false,
  webpack: (webpackConfig) => {
    // For web3modal
    webpackConfig.externals.push("pino-pretty", "lokijs", "encoding")
    return webpackConfig
  },
};

export default nextConfig;
