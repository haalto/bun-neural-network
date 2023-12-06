import { describe, expect, it } from "bun:test";
import { Matrix, matrix } from "mathjs";
import { accuracyHotOneEncoded, categoricalCrossEntropy } from "./loss";

describe("loss functions", () => {
  it("should calculate the categorical cross-entropy", () => {
    const pred = matrix([[0.7, 0.1, 0.2]]);
    const labels = matrix([[1, 0, 0]]);
    const loss = categoricalCrossEntropy(pred, labels);
    expect(loss).toBeCloseTo(0.35667494393873245);
  });

  it("should be able to categorical cross-entropy", () => {
    const pred = matrix([
      [0.7, 0.2, 0.1],
      [0.5, 0.1, 0.4],
      [0.02, 0.9, 0.08],
    ]);
    const labels = matrix([
      [1, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
    ]);
    const loss = categoricalCrossEntropy(pred, labels);
    console.log(loss);
    expect(loss).toBeCloseTo(0.38506088005216804);
  });
});

describe("accuracy", () => {
  it("should calculate the accuracy", () => {
    const pred = matrix([
      [0.7, 0.2, 0.1],
      [0.5, 0.1, 0.4],
      [0.02, 0.9, 0.08],
    ]);
    const labels = matrix([
      [1, 0, 0],
      [1, 0, 0],
      [0, 0, 1],
    ]);
    const accuracy = accuracyHotOneEncoded(pred, labels);
    console.log(accuracy);
    expect(accuracy).toBeCloseTo(0.6666666666666666);
  });
});
