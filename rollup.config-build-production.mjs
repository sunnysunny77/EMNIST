import { babel } from "@rollup/plugin-babel";
import { nodeResolve } from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import terser from "@rollup/plugin-terser";

export default {
  input: "./js/index.js",
  output: [
    {
      file: "./site/js/app.min.js",
      format:  "iife",
      plugins: [terser()],
    },
  ],
  plugins: [
    babel({ babelHelpers: "bundled" }),
    commonjs({
      include: /node_modules/,
    }),
    nodeResolve(),
  ],
};