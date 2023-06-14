import sequtils, strformat
import nimmicrograd

let n = newMLP(3, @[4, 4, 1])
let xs: seq[seq[float]] = @[
    @[2.0, 3.0, -1.0],
    @[3.0, -1.0, 0.5],
    @[0.5, 1.0, 1.0],
    @[1.0, 1.0, -1.0]
]
let ys: seq[float] = @[1.0, -1.0, -1.0, 1.0]

for k in 0..<20:
    # forward pass
    var ypred = newSeq[Value]()
    for x in xs:
        ypred.add(n.forward(x)[0])

    var loss = newValue(0)
    for (ygt, yout) in zip(ys, ypred):
        loss += (yout - ygt)**2

    # backward pass
    n.zeroGrad()
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.005 * p.grad

    echo fmt"{k:2} {loss.data:.15f}"
