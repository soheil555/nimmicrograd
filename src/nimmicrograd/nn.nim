import engine, random, sequtils, strformat, strutils

# Module

type
    Module* = ref object of RootObj

proc parameters*(self: Module): seq[Value] =
    return @[]

proc zeroGrad*(self: Module) =
    for p in self.parameters():
        p.grad = 0

# Neuron

type
    Activation* = enum
        Tanh
        Relu
    Neuron* = ref object of Module
        nin*: int
        weights*: seq[Value]
        bias*: Value
        case activation*: Activation
        of Relu:
            nonlin: bool
        else:
            discard

proc newNeuron*(nin: int, activation = Tanh, nonlin = true): Neuron =
    var weights = newSeq[Value]()
    for _ in 0..<nin:
        weights.add(newValue(gauss()))
    let bias = newValue(gauss())

    result = Neuron(nin: nin, weights: weights, bias: bias, activation: activation)

    if result.activation == Relu:
        result.nonlin = nonlin

proc call*(self: Neuron, x: seq[float|Value]): Value =
    var act = self.bias
    for (wi, xi) in zip(self.weights, x):
        act = act + (wi*xi)

    result = case self.activation
        of Tanh:
            act.tanh()
        of Relu:
            if self.nonlin: act.relu() else: act

proc parameters*(self: Neuron): seq[Value] =
    self.weights & @[self.bias]

proc `$`*(self: Neuron): string =
    let activation = case self.activation
        of Tanh:
            "Tanh"
        of Relu:
            if self.nonlin: "ReLU" else: "Linear"
    result = fmt"{activation} Neuron({self.weights.len})"

# Layer

type
    Layer* = ref object of Module
        nin*: int
        nout*: int
        neurons*: seq[Neuron]

proc newLayer*(nin: int, nout: int, activation = Tanh, nonlin = true): Layer =
    var neurons = newSeq[Neuron]()
    for _ in 0..<nout:
        neurons.add(newNeuron(nin, activation, nonlin))

    return Layer(nin: nin, nout: nout, neurons: neurons)

proc call*(self: Layer, x: seq[float|Value]): seq[Value] =
    result = newSeq[Value]()
    for n in self.neurons:
        result.add(n.call(x))

proc parameters*(self: Layer): seq[Value] =
    result = newSeq[Value]()
    for n in self.neurons:
        result &= n.parameters()

proc `$`*(self: Layer): string =
    let neurons = self.neurons.mapIt($it).join(", ")
    result = fmt"Layer of [{neurons}]"

# MLP

type
    MLP* = ref object of Module
        layers*: seq[Layer]

proc newMLP*(nin: int, nouts: seq[int], activation = Tanh): MLP =
    let sz = @[nin] & nouts
    var layers = newSeq[Layer]()
    for i in 0..<nouts.len:
        layers.add(newLayer(sz[i], sz[i+1], activation, i != nouts.len-1))

    return MLP(layers: layers)

proc call*(self: MLP, x: seq[Value]): seq[Value] =
    var copied = seq(x)
    for layer in self.layers:
        copied = layer.call(copied)

    return copied

proc call*(self: MLP, x: seq[float]): seq[Value] =
    return self.call(x.mapIt(newValue(it)))

proc parameters*(self: MLP): seq[Value] =
    result = newSeq[Value]()
    for layer in self.layers:
        result &= layer.parameters()

proc `$`*(self: MLP): string =
    let layers = self.layers.mapIt($it).join(", ")
    result = fmt"MLP of [{layers}]"
