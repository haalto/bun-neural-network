import { Matrix } from "mathjs";
import { Layer } from "../layers/Layer";

export class NeuralNetwork {
  private layers: Layer[] = [];

  addLayer(layer: Layer) {
    this.layers.push(layer);
  }

  train(x: Matrix) {
    const output = this.layers.reduce(
      (input, layer) => layer.forward(input),
      x
    );
    return output;
  }
}
