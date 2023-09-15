/**
 * File              : lib.rs
 * Author            : Yue Peng <yuepaang@gmail.com>
 * Date              : 14.09.2023
 * Last Modified Date: 14.09.2023
 * Last Modified By  : Yue Peng <yuepaang@gmail.com>
 */
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet},
    hash::Hash,
    iter::{from_fn, FromIterator},
};

use pyo3::prelude::*;
use std::collections::HashMap;

use indexmap::IndexSet;

use petgraph::{
    prelude::*,
    visit::{IntoNeighborsDirected, NodeCount},
    Direction::Outgoing,
};

use itertools::Itertools;

use rayon::prelude::*;

pub fn all_simple_paths<TargetColl, G, IsGoal, EarlyStop>(
    graph: G,
    from: G::NodeId,
    is_goal: IsGoal,
    early_stop: EarlyStop,
) -> impl Iterator<Item = TargetColl>
where
    G: NodeCount,
    G: IntoNeighborsDirected,
    G::NodeId: Eq + Hash,
    IsGoal: Fn(&G::NodeId) -> bool,
    EarlyStop: Fn(&G::NodeId) -> bool,
    TargetColl: FromIterator<G::NodeId>,
{
    // list of visited nodes
    let mut visited: IndexSet<G::NodeId> = IndexSet::from_iter(Some(from));
    // list of childs of currently exploring path nodes,
    // last elem is list of childs of last visited node
    let mut stack = vec![graph.neighbors_directed(from, Outgoing)];

    from_fn(move || {
        while let Some(children) = stack.last_mut() {
            if let Some(child) = children.next() {
                if early_stop(&child) {
                    continue;
                }
                if is_goal(&child) {
                    let path = visited
                        .iter()
                        .cloned()
                        .chain(Some(child))
                        .collect::<TargetColl>();
                    return Some(path);
                } else if !visited.contains(&child) {
                    visited.insert(child);
                    stack.push(graph.neighbors_directed(child, Outgoing));
                }
            } else {
                stack.pop();
                visited.pop();
            }
        }
        None
    })
}

#[pyclass]
struct Graph {
    g: DiGraph<i32, i32>,
    nodes: HashMap<i32, NodeIndex>,
}

#[pymethods]
impl Graph {
    #[new]
    fn from_arcs(arcs: Vec<(i32, i32)>) -> PyResult<Self> {
        let mut g: DiGraph<i32, i32> = DiGraph::new();
        let it = arcs
            .iter()
            .map(|x| &x.0)
            .chain(arcs.iter().map(|x| &x.1))
            .unique();
        let nodes: HashMap<_, _> = it.map(|x| (x.clone(), g.add_node(x.clone()))).collect();

        for (o, d) in arcs {
            g.add_edge(nodes[&o], nodes[&d], 0);
        }

        Ok(Graph { g, nodes })
    }

    fn is_connected(&self, from: i32, to: i32) -> bool {
        let it = all_simple_paths(
            &self.g,
            self.nodes[&from],
            |n| &self.g[*n] == &to,
            |_| false,
        );
        let path: Vec<Vec<i32>> = it
            .map(|v: Vec<_>| v.into_iter().map(|i| &self.g[i]).cloned().collect())
            .take(1)
            .collect();
        if path.is_empty() {
            false
        } else {
            true
        }
    }

    fn co_gen_paths(&self, k: usize, ods: Vec<(i32, i32)>) -> Vec<Vec<Vec<i32>>> {
        ods.par_iter()
            .map(|od| {
                let (from, to) = od;

                // let g = EdgeFiltered::from_fn(&self.g, |e| {
                //     if self.g[e.source()] == self.g[e.target()] {
                //         true
                //     } else {
                //         false
                //     }
                // });
                let it =
                    all_simple_paths(&self.g, self.nodes[&from], |n| &self.g[*n] == to, |_| false);
                it.map(|v: Vec<_>| v.into_iter().map(|i| &self.g[i]).cloned().collect())
                    .take(k)
                    .collect()
            })
            .collect()
    }
}

type Point = (i32, i32);

fn distance(p1: Point, p2: Point) -> usize {
    (p1.0.abs_diff(p2.0) + p1.1.abs_diff(p2.1)) as usize
}

#[derive(Clone, Eq, PartialEq)]
struct Node {
    cost: usize,
    point: Point,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl ToString for Direction {
    fn to_string(&self) -> String {
        match self {
            Direction::Up => "UP".to_string(),
            Direction::Down => "DOWN".to_string(),
            Direction::Left => "LEFT".to_string(),
            Direction::Right => "RIGHT".to_string(),
        }
    }
}

fn astar(start: Point, end: Point, blocked: HashSet<Point>) -> Option<String> {
    let mut to_visit = BinaryHeap::new();
    to_visit.push(Node {
        cost: 0,
        point: start,
    });

    let mut visited = HashSet::new();
    let mut g_scores = HashMap::new();
    g_scores.insert(start, 0);

    let mut came_from = HashMap::new();

    while let Some(Node { cost: _, point }) = to_visit.pop() {
        if point == end {
            let mut path: Vec<Direction> = Vec::new();
            let mut current = end;

            while let Some(&(direction, parent)) = came_from.get(&current) {
                path.push(direction);
                current = parent;
            }

            path.reverse();
            return Some(path.get(0).unwrap().to_string());
        }

        for &(i, j, direction) in &[
            (-1, 0, Direction::Left),
            (1, 0, Direction::Right),
            (0, -1, Direction::Down),
            (0, 1, Direction::Up),
        ] {
            let next = (point.0 + i, point.1 + j);

            if visited.contains(&next) || blocked.contains(&next) {
                continue;
            }

            let tentative_g_score = g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&usize::MAX) {
                came_from.insert(next, (direction, point));
                g_scores.insert(next, tentative_g_score);
                let f_score = tentative_g_score + distance(next, end);
                to_visit.push(Node {
                    cost: f_score,
                    point: next,
                });
            }
        }

        visited.insert(point);
    }

    None
}

fn astar_path(start: Point, end: Point, blocked: HashSet<Point>) -> Option<(usize, Vec<String>)> {
    let mut to_visit = BinaryHeap::new();
    to_visit.push(Node {
        cost: 0,
        point: start,
    });

    let mut visited = HashSet::new();
    let mut g_scores = HashMap::new();
    g_scores.insert(start, 0);

    let mut came_from = HashMap::new();

    while let Some(Node { cost, point }) = to_visit.pop() {
        if point == end {
            let mut path: Vec<Direction> = Vec::new();
            let mut current = end;

            while let Some(&(direction, parent)) = came_from.get(&current) {
                path.push(direction);
                current = parent;
            }

            path.reverse();
            let direction_path = path.into_iter().map(|x| x.to_string()).collect();
            return Some((cost, direction_path));
        }

        for &(i, j, direction) in &[
            (-1, 0, Direction::Left),
            (1, 0, Direction::Right),
            (0, -1, Direction::Down),
            (0, 1, Direction::Up),
        ] {
            let next = (point.0 + i, point.1 + j);

            if visited.contains(&next) || blocked.contains(&next) {
                continue;
            }

            let tentative_g_score = g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&usize::MAX) {
                came_from.insert(next, (direction, point));
                g_scores.insert(next, tentative_g_score);
                let f_score = tentative_g_score + distance(next, end);
                to_visit.push(Node {
                    cost: f_score,
                    point: next,
                });
            }
        }

        visited.insert(point);
    }

    None
}

#[pyfunction]
fn get_direction(start: Point, end: Point, blocked: Vec<Point>) -> PyResult<String> {
    let blocked: HashSet<Point> = blocked.into_iter().collect();
    let direction = astar(start, end, blocked).unwrap_or("STAY".to_string());
    Ok(direction)
}

#[pyfunction]
fn get_direction_path(
    start: Point,
    end: Point,
    blocked: Vec<Point>,
) -> PyResult<(usize, Vec<String>)> {
    let blocked: HashSet<Point> = blocked.into_iter().collect();
    let direction_path = astar_path(start, end, blocked).unwrap();
    Ok(direction_path)
}

#[pymodule]
fn rust_perf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Graph>()?;
    m.add_function(wrap_pyfunction!(get_direction, m)?)?;
    m.add_function(wrap_pyfunction!(get_direction_path, m)?)?;
    Ok(())
}

