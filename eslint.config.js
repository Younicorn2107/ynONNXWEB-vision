import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/", "node_modules/", "coverage/"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    rules: {
      // `_name` marks a parameter the signature needs and the body does not.
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_", caughtErrors: "none" },
      ],
      "@typescript-eslint/consistent-type-imports": ["error", { fixStyle: "inline-type-imports" }],
      eqeqeq: ["error", "always"],
      "no-console": "error",
      // Text only (security review, E.17 rule 1): no string is ever parsed as
      // HTML. Pages build nodes (textContent, new Option, replaceChildren);
      // a library returns text. The demo's one reviewed exception is
      // `markup` in src/dom.ts (static source text only).
      "no-restricted-properties": [
        "error",
        ...["innerHTML", "outerHTML", "insertAdjacentHTML", "createContextualFragment", "srcdoc"].map((property) => ({
          property,
          message: "Build nodes (textContent, new Option, replaceChildren): no string is parsed as HTML.",
        })),
        ...["write", "writeln"].map((property) => ({
          object: "document",
          property,
          message: "Build nodes (textContent, new Option, replaceChildren): no string is parsed as HTML.",
        })),
      ],
    },
  },
);
