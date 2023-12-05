import { matrix } from "mathjs";
import { FCLayer } from "./layers/FCLayer";
import { NeuralNetwork } from "./network/NeuralNetwork";
import { ReLU } from "./layers/activation/ReLU";
import { Softmax } from "./layers/activation/Softmax";

const input = matrix([
  [1, 2, 3, 2.5],
  [2.0, 5.0, -1.0, 2.0],
  [-1.5, 2.7, 3.3, -0.8],
]);

const neuralNetwork = new NeuralNetwork();

neuralNetwork.addLayer(new FCLayer(4, 5));
neuralNetwork.addLayer(new ReLU());
neuralNetwork.addLayer(new FCLayer(5, 2));
neuralNetwork.addLayer(new Softmax());
const output = neuralNetwork.train(input);
console.log(output.toArray());
