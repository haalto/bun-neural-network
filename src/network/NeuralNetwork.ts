import { Matrix, arg, max } from "mathjs";
import { Layer } from "../layers/Layer";
import { Loss } from "../loss/Loss";
import { accuracyHotOneEncoded } from "../math/loss";

export class NeuralNetwork {
  private layers: Layer[] = [];
  private loss: Loss;

  constructor(lossFunction: Loss) {
    this.loss = lossFunction;
  }

  addLayer(layer: Layer) {
    this.layers.push(layer);
  }

  addLoss(loss: Loss) {
    this.loss = loss;
  }

  train(train: Matrix, labels: Matrix) {
    const output = this.layers.reduce(
      (input, layer) => layer.forward(input),
      train
    );
    const lossValue = this.loss.calculate(output, labels);
    const accuracy = accuracyHotOneEncoded(output, labels);
    return { output, lossValue, accuracy };
  }
}
