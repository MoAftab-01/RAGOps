import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

/**
 * Flat config for the console.
 *
 * Deliberately narrow. The rules here are the ones that catch a real class of
 * defect in this codebase, not a "looks impressive" pile:
 *
 * - `no-unnecessary-condition` and `strict-boolean-expressions` are the
 *   machinery behind this project's central rule. A value that is *always*
 *   defined, or a ternary whose branches are identical, is either dead code or
 *   a check that can never fire — and a check that can never fire is a claim
 *   the UI makes that it cannot keep. Both were found by hand in this codebase
 *   (`changeTone`'s `change === 0 ? "flat" : "flat"`), which is exactly why
 *   they are rules here rather than a review habit.
 * - Type-aware linting needs `projectService`, which is why the config is
 *   `tseslint.configs.recommendedTypeChecked` rather than the untyped variant.
 */
export default tseslint.config(
  {
    ignores: ["dist/**", "node_modules/**", "coverage/**"],
  },

  js.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,
  reactHooks.configs["recommended-latest"],

  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      globals: { ...globals.browser, ...globals.es2022 },
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      /* ---------------------------------------------------------------- */
      /* Honesty rules                                                      */
      /* ---------------------------------------------------------------- */

      // A conditional that cannot fire is either dead code or a promise the UI
      // cannot keep. Both are bugs in a console whose job is to be trustworthy.
      "@typescript-eslint/no-unnecessary-condition": "error",

      // `"ok"` is a string; `ok` is not the same value. Forcing the explicit
      // comparison stops `status && <Badge/>` rendering `""` into the DOM.
      //
      // `allowNullableString` is set because this project's schemas are full of
      // `field: string | null`, and `x !== null` is a longer way of writing
      // `Boolean(x)` for a value the API already told us is absent. It does not
      // re-admit `""`: an empty string still needs `!== ""`, which is why
      // `ApplicationsPage` tests both.
      "@typescript-eslint/strict-boolean-expressions": [
        "error",
        {
          allowString: true,
          allowNumber: false,
          allowNullableObject: true,
          allowNullableString: true,
        },
      ],

      /* ---------------------------------------------------------------- */
      /* Correctness                                                       */
      /* ---------------------------------------------------------------- */

      // An unhandled promise in a click handler is a silently swallowed 401.
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/no-misused-promises": [
        "error",
        { checksVoidReturn: { attributes: false } },
      ],

      // `e` in a catch is the only place a reason can be read; dropping it is
      // how "something went wrong" replaces the backend's own message.
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_", caughtErrors: "none" },
      ],
      "no-unused-vars": "off",

      // `ReactNode` in state is the classic source of a crash on render.
      "@typescript-eslint/no-confusing-void-expression": [
        "error",
        { ignoreArrowShorthand: true },
      ],

      // An unescaped regex or a placeholder built by concatenation is where a
      // search box turns into an injection. Cheap to require, rarely painful.
      "no-useless-escape": "error",
      eqeqeq: ["error", "always", { null: "ignore" }],
    },
  },

  {
    // Tests assert on rendered text, so long literals, non-null assertions after
    // an explicit query, and untyped JSON bodies are the normal shape rather
    // than a smell. `require-await` is off for the same reason: a `fetch` stub
    // has to be async to be a `fetch`, even when the body is a constant — the
    // alternative is a hand-rolled thenable, which tests nothing real.
    files: ["**/*.{test,spec}.{ts,tsx}", "src/test/**/*.{ts,tsx}"],
    rules: {
      "@typescript-eslint/no-non-null-assertion": "off",
      "@typescript-eslint/no-unnecessary-condition": "off",
      "@typescript-eslint/no-unsafe-assignment": "off",
      "@typescript-eslint/no-unsafe-member-access": "off",
      "@typescript-eslint/no-base-to-string": "off",
      "@typescript-eslint/require-await": "off",
      // Asserting a partial stand-in onto a global is the whole job of a jsdom
      // shim — the alternative is the test environment simply lacking the API.
      "@typescript-eslint/no-unnecessary-type-assertion": "off",
    },
  },

  {
    // Config files are plain JS with no project to attach to.
    files: ["*.config.{js,ts}", "eslint.config.js"],
    ...tseslint.configs.disableTypeChecked,
  },
);
