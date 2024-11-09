# Scatter Plot of Investment Transactions

The current time is ${new Date(now).toLocaleTimeString("en-US")}.

## Import Open Investment Data

```js
import {FileAttachment} from "npm:@observablehq/stdlib";

const data = FileAttachment("./data/OID-Dataset.csv").csv({typed: true});
```

```js
display(data)
```

## Display as table
```js
Inputs.table(data)
```

## Plot
```js
// function scatterPlot(data, {width} = {}) {
//     return Plot.plot({
//         marginLeft: 50,
//         width,
//         inset: 10,
//         grid: true,
//         color: {
//             legend: true,
//         },
//         y: {type: "symlog"},
//         marks: [
//             Plot.dot(data, {
//                 x: "Transaction Date", 
//                 y: "No. of shares", 
//                 stroke: "Action",
//                 channels: {Ticker: "Ticker",Name: "Name" },
//                 tip: true
//                 })
//         ]
//     })
// }

Plot.plot({
    marginLeft: 50,
    width,
    inset: 10,
    grid: true,
    color: {
        legend: true,
    },
    y: {type: "symlog"},
    marks: [
        Plot.dot(data, {
            x: "Transaction Date", 
            y: "Price / share", 
            stroke: "Action",
            channels: {Ticker: "Ticker",Name: "Name" },
            tip: true
            })
    ]
})
```

<!-- 
<div class="grid grid-cols-1">
  <div class="card">${resize((width) => scatterPlot(data, {width}))}</div>
</div> -->