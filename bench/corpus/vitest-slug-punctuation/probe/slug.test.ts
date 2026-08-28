import { describe, expect, it } from "vitest";

import { slugify } from "../../../src/slug";

describe("slugify", () => {
  it("normalizes punctuation as separators", () => {
    expect(slugify("Hello, world!")).toBe("hello-world");
  });
});

