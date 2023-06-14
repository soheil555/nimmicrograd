import strformat, algorithm, math

type
    Operation* = enum
        None = ""
        Add = "+"
        Mul = "*"
        Pow = "**"
        Tanh = "tanh"
        Relu = "relu"
        Exp = "exp"

    Value* = ref object
        data*: float
        grad*: float
        label*: string
        children*: seq[Value]
        case op*: Operation
        of Pow:
        exponent: float
        else:
            discard

proc newValue*(data: float, children = newSeq[Value](), op = None, label = "",
        exponent: float = 1): Value =
    result = Value(data: data, grad: 0, children: children, op: op, label: label)
    if op == Pow:
        result.exponent = exponent

# add

proc `+`*(self: Value, other: Value): Value =
    newValue(self.data + other.data, @[self, other], Add)

proc `+`*(self: Value, other: float): Value =
    self + newValue(other)

proc `+`*(self: float, other: Value): Value =
    other + self

# mul

proc `*`*(self: Value, other: Value): Value =
    newValue(self.data * other.data, @[self, other], Mul)

proc `*`*(self: Value, other: float): Value =
    self * newValue(other)

proc `*`*(self: float, other: Value): Value =
    other * self

# Sub

proc `-`*(self: Value): Value =
    self * -1

proc `-`*(self: Value, other: Value): Value =
    self + (-other)

proc `-`*(self: Value, other: float): Value =
    self + (-other)

proc `-`*(self: float, other: Value): Value =
    -other + self

# pow

proc `**`*(self: Value, other: float): Value =
    newValue(self.data.pow(other), @[self], Pow, exponent = other)

# div

proc `/`*(self: Value, other: Value): Value =
    self * (other ** -1)

proc `/`*(self: Value, other: float): Value =
    self * (other.pow(-1))

proc `/`*(self: float, other: Value): Value =
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
            v.children[0].grad += v.exponent * v.children[0].data.pow(
                    v.exponent-1) * v.grad
        of Tanh:
            v.children[0].grad += (1 - v.data.pow(2)) * v.grad
        of Relu:
            v.children[0].grad += (v.data > 0).float * v.grad
        of Exp:
            v.children[0].grad += v.data * v.grad
        of None:
            discard

proc `$`*(self: Value): string =
    fmt"Value(data={self.data}, grad={self.grad})"
