# Scatter Plot of Investment Transactions

The current time is ${new Date(now).toLocaleTimeString("en-US")}.

## Import Open Investment Data

```js
import {FileAttachment} from "npm:@observablehq/stdlib";
import * as vega from "npm:vega";
import * as vegaLite from "npm:vega-lite";
import * as vegaLiteApi from "npm:vega-lite-api";
import * as tooltip from "npm:vega-tooltip";
const vl = vegaLiteApi.register(vega, vegaLite, {
  init: (view) => {
    view.tooltip(new tooltip.Handler().call);

    // another suggestion from https://observablehq.com/@vega/vega-lite-api-v5#vl
    if (view.container()) view.container().style["overflow-x"] = "auto";
  }
});

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

vl.markPoint({filled: true})
    .data(data)
    .params(
        vl.selectInterval().bind('scales')
    )
    .encode(
        vl.x().fieldT('Transaction Date'),
        vl.y().fieldQ('Price / share').scale({type: 'log', domain: [0.001, 100]}),
        vl.color().fieldN('Action'),
        vl.shape().fieldN('Action'),
        vl.tooltip(['Ticker', 'Name', 'Price / share'])
    )
    .width(1000)
    .height(600)
    .render()

// {
//   const brush = vl.selectInterval().encodings('x');
//   const x = vl.x().fieldT('Transaction Date').title(null);
  
//   const base = vl.markArea()
//     .encode(x, vl.y().fieldQ('Price / share'))
//     .width(700);
  
//   return vl.data(data)
//     .vconcat(
//       base.encode(x.scale({domain: brush})),
//       base.params(brush).height(60)
//     )
//     .render();
// }


```

<!-- 
<div class="grid grid-cols-1">
  <div class="card">${resize((width) => scatterPlot(data, {width}))}</div>
</div> -->