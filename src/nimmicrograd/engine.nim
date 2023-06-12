import strformat, algorithm, math

type 
    Operation = enum
        None = ""
        Add = "+"
        Mul = "*"
        Pow = "**"
        Tanh = "tanh"
        Relu = "relu"
        Exp = "exp"

    Value* = ref object
        data*: float64
        grad*: float64
        label*: string
        children: seq[Value]
        op: Operation
    
proc newValue*(data: float64, children = newSeq[Value](), op = None, label = ""): Value =
    Value(data: data, grad: 0, children: children, op: op, label: label)

# add

proc `+`*(self: Value, other: Value): Value =
    newValue(self.data + other.data, @[self, other], Add)

proc `+`*(self: Value, other: float64): Value =
    self + newValue(other)

proc `+`*(self: float64, other: Value): Value =
    other + self

# mul

proc `*`*(self: Value, other: Value): Value =
    newValue(self.data * other.data, @[self, other], Mul)

proc `*`*(self: Value, other: float64): Value =
    self * newValue(other)

proc `*`*(self: float64, other: Value): Value =
    other * self

# Sub

proc `-`*(self: Value): Value =
    self * -1

proc `-`*(self: Value, other: Value): Value =
    self + (-other)

proc `-`*(self: Value, other: float64): Value =
    self + (-other)

proc `-`*(self: float64, other: Value): Value =
    -other + self

# pow

proc `**`*(self: Value, other: float64): Value =
    newValue(self.data.pow(other), @[self], Pow)

# div

proc `/`*(self: Value, other: Value): Value =
    self * (other ** -1)

proc `/`*(self: Value, other: float64): Value =
    self * (other.pow(-1))

proc `/`*(self: float64, other: Value): Value =
    self * (other ** -1)

proc tanh*(self: Value): Value =
    let x = self.data
    let expTwoX = exp(2*x)
    let t = (expTwoX - 1) / (expTwoX + 1)
    result = newValue(t, @[self], Tanh)

proc relu*(self: Value): Value =
    newValue(if self.data < 0: 0.0 else: self.data, @[self], Relu)

proc exp*(self: Value): Value =
    newValue(self.data.exp(), @[self], Exp)
    
proc backward*(self: Value): void =
    var topo = newSeq[Value]()
    var visited = newSeq[Value]()

    proc buildTopo (v: Value) =
        if v notin visited:
            visited.add(v)
            for child in v.children:
                buildTopo(child)
            topo.add(v)
    buildTopo(self)

    self.grad = 1

    for v in topo.reversed():
        case v.op
        of Add:
            v.children[0].grad += v.grad
            v.children[1].grad += v.grad
        of Mul:
            v.children[0].grad += v.children[1].data * v.grad
            v.children[1].grad += v.children[0].data * v.grad
        of Pow:
            var child = v.children[0]
            let exponent = v.data.log10() / child.data.log10()
            child.grad += exponent * v.data / child.data * v.grad
        of Tanh:
            let t = v.data
            v.children[0].grad += (1 - t.pow(2)) * v.grad
        of Relu:
            v.children[0].grad += (v.data > 0).float64 * v.grad
        of Exp:
            v.children[0].grad += v.data * v.grad
        of None:
            discard

proc `$`*(self: Value): string =
    fmt"Value(data={self.data}, grad={self.grad})"