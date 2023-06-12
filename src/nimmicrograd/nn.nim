import engine, random, sequtils

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
    Neuron* = ref object of Module
        nin*: int
        weights*: seq[Value]
        bias*: Value


proc newNeuron*(nin: int): Neuron =
    var weights = newSeq[Value]()
    for _ in 0..<nin:
        weights.add(newValue(gauss()))
    let bias = newValue(gauss())

    return Neuron(nin: nin, weights: weights, bias: bias)

proc call*(self: Neuron, x: seq[float|Value]): Value =
    var act = self.bias
    for (wi, xi) in zip(self.weights, x):
        act = act + (wi*xi)
    
    result = act.tanh()
    
proc parameters*(self: Neuron): seq[Value] =
    self.weights & @[self.bias]
    
# Layer

type
    Layer* = ref object of Module
        nin*: int
        nout*: int
        neurons*: seq[Neuron]
    
proc newLayer*(nin: int, nout: int): Layer =
    var neurons = newSeq[Neuron]()
    for _ in 0..<nout:
        neurons.add(newNeuron(nin))
    
    return Layer(nin: nin, nout: nout, neurons: neurons)

proc call*(self: Layer, x: seq[float|Value]): seq[Value] =
    result = newSeq[Value]()
    for n in self.neurons:
        result.add(n.call(x))
    
proc parameters*(self: Layer): seq[Value] =
    result = newSeq[Value]()
    for n in self.neurons:
        result &= n.parameters()

# MLP

type
    MLP* = ref object of Module
        layers*: seq[Layer]

proc newMLP*(nin: int, nouts: seq[int]): MLP =
    let sz = @[nin] & nouts
    var layers = newSeq[Layer]()
    for i in 0..<nouts.len:
        layers.add(newLayer(sz[i], sz[i+1]))
    
    return MLP(layers: layers)

proc call*(self: MLP, x: seq[float|Value]) =
    for layer in self.layers:
        x = layer.call(x)

    return x

proc parameters*(self: MLP): seq[Value] =
    result = newSeq[Value]()
    for layer in self.layers:
        result &= layer.parameters()