import { FlatCompat } from "@eslint/eslintrc";

// eslint-config-next still ships an eslintrc-style config; FlatCompat is the
// supported bridge to ESLint 9's flat config (same shape create-next-app emits).
const compat = new FlatCompat({ baseDirectory: import.meta.dirname });

const config = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  { ignores: [".next/**", "node_modules/**", "next-env.d.ts"] },
];

export default config;
