# Project MoirAI - Crime Prediction

The goal of Project MoirAI is to find and design reliable models suitable for processing a large number of different feature categories, most prominently spatio-temporal-linguistic data.

<img src="./images/ui-overview.png">

## Quickstart

1. Clone this repository `git clone https://github.com/project-moirai/crime-prediction.git`
2. Switch to the cloned directory and open `index.html` in a browser. Ensure you can navigate through the map and set filters.
3. Switch directory to `/experiments` and open file `train.py` in an editor. Make sure you have Python installed and run `pip install -r requirements.txt`
4. TBD

## Data format

| Property   | Description                                 | Sample value                         |
|------------|---------------------------------------------|--------------------------------------|
| guid       | Unique Id of the crime event                | aad6ff60-1991-4eea-92cc-75401fc1794c |
| country    | ISO 3166-1 country code                     | de                                   |
| date       | Date of the event in MM-DD format           | 12-31                                |
| time       | Time of the day when the event happened     | 14:30                                |
| category   | Type(s) of incident (see list below)        | ["theft", "drug-offense"]            |
| summary    | Short description of the incident           | Theft of cash and drugs              |
| lat        | Latitude coordinate of the event            | 49.3791                              |
| lng        | Longitude coordinate of the event           | 8.0783                               |

Categories can be at maximum two of the following:

```
["alcohol-related-incident", "animal-incident", "arson", "assault", "criminal-damage", "cybercrime", "domestic-violence", "drug-offense", "environmental-offense", "fraud", "gun-offense", "human-trafficking", "intrusion", "murder", "mountain-accident", "other", "public-disturbance", "robbery", "search-and-rescue", "sexual-assault", "suicide", "traffic-accident", "theft", "vandalism"]
```

Data for each country was collected from the following sources:

| Country    | Source                            |
|------------|-----------------------------------|
| at         | www.polizei.gv.at/{county}/presse |
| de         | www.presseportal.de               |

## Model architecture

TBD

## Sources

## Sources

1. <a name="source-1"></a> Bodnar, C., ... Perdikaris, P. (2024). *A Foundation Model for the Earth System*. arXiv:2405.13063. [https://arxiv.org/abs/2405.13063](https://arxiv.org/abs/2405.13063)
2. <a name="source-2"></a> Abramson, J., ... Jumper, J. M. (2024). *Accurate structure prediction of biomolecular interactions with AlphaFold 3*. Nature, 630, 493–500. [https://www.nature.com/articles/s41586-024-07487-w](https://www.nature.com/articles/s41586-024-07487-w)
3. <a name="source-3"></a> Liu, L., ... Shen, Y. (2024). *How Can Large Language Models Understand Spatial-Temporal Data?* arXiv:2401.14192. [https://arxiv.org/abs/2401.14192](https://arxiv.org/abs/2401.14192)
4. <a name="source-4"></a> Ali, K., ... Fischer, A. (2025). *Enhancing Spatiotemporal Networks with xLSTM: A Scalar LSTM Approach for Cellular Traffic Forecasting*. arXiv:2507.19513. [https://arxiv.org/abs/2507.19513](https://arxiv.org/abs/2507.19513)
