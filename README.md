# GeneticAlgorithmTSP

## Table of contents
- [General info](#general-info)
- [Technologies](#technologies)
- [Description](#description)
  - [Benchmarks](#benchmarks)

## General info
This academic project was completed as the "final project" for the class of 
[Genetic Algorithms and Evolutionary Computing](https://onderwijsaanbod.kuleuven.be/syllabi/e/H02D1AE.htm#activetab=doelstellingen_idp42501408)  at KU Leuven.  

## Technologies
This project was written with: 
- Python 3.8

## Description

The objective of this algorithm is to find the shortest possible tour
for a variable number of cities. The tour_.csv files contain the distance
matrix providing the distance between two given cities, and the number in 
_ provides the total number of cities in the tour. 

My implementation of the genetic algorithm (GA) combines the fundamentals
building blocks of **initialization**, **selection**, **variation**,
**mutation** and **elimination** with heuristics such as the _nearest neighbor_
for a portion of the **initialization** and a modification of the _2-opt_ for 
different stages in the algorithm. 

In a nutshell, the flowchart below summarizes my implementation and for more details regarding, please take a look at the project [report](https://github.com/ymmath/GeneticAlgorithmTSP/blob/main/ProjectReport.pdf):

![](C:\Users\yash_\Downloads\Flowcharts.png)


### Benchmarks

To compare how this implementation of the GA fares (performance results in the [report](https://github.com/ymmath/GeneticAlgorithmTSP/blob/main/ProjectReport.pdf)), a greedy heuristic is applied for each of the tours with the results shown below:
1. **tour29**: simple greedy heuristic 169977 
2. **tour100**: simple greedy heuristic 42655 
3. **tour250**: simple greedy heuristic 49852 
4. **tour500**: simple greedy heuristic 103736 
5. **tour750**: simple greedy heuristic 52453 
6. **tour1000**: simple greedy heuristic 235731