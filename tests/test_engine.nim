import unittest
import nimmicrograd/engine
import math

suite "Value":

  test "Initialization":
    let a = newValue(5)
    let b = newValue(4, @[a], label = "b")

    check:
      b.data == 4
      b.children.len == 1
      b.children[0] == a
      b.op == None
      b.label == "b"
      b.grad == 0

  test "Add":
    let a = newValue(5)
    let b = newValue(4)
    let c = a+b

    check:
      c.data == 9
      c.op == Add
      c.children.len == 2
      c.children[0] == a
      c.children[1] == b

      (a+6).data == 11
      (9+a).data == 14

  test "Sub":
    let a = newValue(5)
    let b = newValue(4)
    let c = a-b

    check:
      c.data == 1

      (-a).data == -5
      (a-6).data == -1
      (9-a).data == 4

  test "Mul":
    let a = newValue(5)
    let b = newValue(4)
    let c = a*b

    check:
      c.data == 20
      c.op == Mul
      c.children.len == 2
      c.children[0] == a
      c.children[1] == b

      (a*2).data == 10
      (3*a).data == 15
      (4 * -a).data == -20

  test "Pow":
    let a = newValue(5)
    let b = a**2

    check:
      b.data == 25
      b.op == Pow
      b.children.len == 1
      b.children[0] == a

  test "Div":
    let a = newValue(5)
    let b = newValue(4)
    let c = a/b

    check:
      c.data == 5/4

      (a/5).data == 1
      (10/a).data == 2

  test "Tanh":
    let a = newValue(5)
    let b = a.tanh()

    check:
      almostEqual(b.data, tanh(5.0))
      b.op == Tanh
      b.children.len == 1
      b.children[0] == a

  test "Relu":
    let a = newValue(5)
    let b = newValue(-1)
    let c = a.relu()
    let d = b.relu()

    check:
      c.data == 5
      c.op == Relu
      c.children.len == 1
      c.children[0] == a

      d.data == 0

  test "Exp":
    let a = newValue(5)
    let b = a.exp()

    check:
      b.data == exp(5.0)
      b.op == Exp
      b.children.len == 1
      b.children[0] == a

  test "Backward":
    var a = newValue(-4)
    var b = newValue(2)
    var c = a + b
    var d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).tanh()
    d += d.exp()
    var e = c - d
    var f = e**2
    var g = f / 2
    g += 10 / f

    g.backward()

    check:
      almostEqual(g.data, 11.58007706124438)
      almostEqual(a.grad, 390.47003923590023)
      almostEqual(b.grad, 1598.1048305462182)

  test "$":
    let a = newValue(5)

    check:
      $a == "Value(data=5.0, grad=0.0)"



