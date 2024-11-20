---
sql: 
    investmentTransaction: ./data/OID-Dataset.csv
---

# Scatter Plot of Investment Transactions


```sql id=data display
SELECT * FROM investmentTransaction
```

```js
const investmentPlot = Plot.plot({
        marginLeft: 50,
        inset: 10,
        grid: true,
        color: {
            legend: true,
        },
        y: {type: "symlog"},
        marks: [
            Plot.dot(data, {
                x: "Transaction Date", 
                y: "No. of shares", 
                stroke: "Action",
                channels: {Ticker: "Ticker",Name: "Name" },
                tip: true
                })
        ]
    });

display(investmentPlot);
```