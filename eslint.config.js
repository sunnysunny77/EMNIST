import globals from "globals";
import js from "@eslint/js";

export default [
  js.configs.recommended,
  {
    rules: {
      semi: ["warn", "always"],
      quotes: ["error", "double"]
    },
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: {
        ...globals.browser,
      },
    },
  }
];
