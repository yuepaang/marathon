/**
 * File              : lib.rs
 * Author            : Yue Peng <yuepaang@gmail.com>
 * Date              : 14.09.2023
 * Last Modified Date: 26.09.2023
 * Last Modified By  : Yue Peng <yuepaang@gmail.com>
 */
use std::{
    cmp::Ordering,
    collections::HashMap,
    collections::HashSet,
    sync::{Arc, Mutex},
};

use algo::bfs;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::algo::deal_with_enemy_nearby;

mod algo;
mod conf;

pub type Point = (i32, i32);

pub fn distance(p1: Point, p2: Point) -> i32 {
    (p1.0.abs_diff(p2.0) + p1.1.abs_diff(p2.1)) as i32
}

#[derive(Clone, Eq, PartialEq)]
pub struct Node {
    cost: i32,
    point: Point,
    passwall: usize,
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
pub enum Direction {
    Stay,
    Up,
    Down,
    Left,
    Right,
}

impl ToString for Direction {
    fn to_string(&self) -> String {
        match self {
            Direction::Stay => "STAY".to_string(),
            Direction::Up => "UP".to_string(),
            Direction::Down => "DOWN".to_string(),
            Direction::Left => "LEFT".to_string(),
            Direction::Right => "RIGHT".to_string(),
        }
    }
}
impl Direction {
    fn get_next_pos(&self, pos: Point) -> Point {
        match self {
            Direction::Stay => (pos.0, pos.1),
            Direction::Up => (pos.0, pos.1 - 1),
            Direction::Down => (pos.0, pos.1 + 1),
            Direction::Left => (pos.0 - 1, pos.1),
            Direction::Right => (pos.0 + 1, pos.1),
        }
    }
}

#[pyfunction]
fn get_direction(start: Point, end: Point, blocked: Vec<Point>) -> PyResult<String> {
    let blocked: HashSet<Point> = blocked.into_iter().collect();
    let direction = algo::astar(start, end, blocked).unwrap_or("STAY".to_string());
    Ok(direction)
}

#[pyfunction]
fn get_direction_path(
    start: Point,
    end: Point,
    blocked: Vec<Point>,
) -> PyResult<(i32, Vec<String>)> {
    let blocked: HashSet<Point> = blocked.into_iter().collect();
    let direction_path = algo::astar_path(start, end, blocked).unwrap();
    Ok(direction_path)
}

#[pyfunction]
fn check_stay_or_not(
    start: Point,
    enemies_position: Vec<Point>,
    pass_wall: usize,
) -> PyResult<String> {
    let banned_points: HashSet<_> = conf::WALLS.iter().cloned().collect();

    // Calculate the overall direction if the cell is further away from all enemies
    for &(i, j, direction) in &[
        (-1, 0, "LEFT"),
        (1, 0, "RIGHT"),
        (0, 1, "DOWN"),
        (0, -1, "UP"),
    ] {
        let mut all_away = true;
        let next = (start.0 + i, start.1 + j);
        if algo::check_out_of_bound(next) {
            continue;
        }

        // PORTAL MOVE
        // let index = conf::PORTALS
        //     .iter()
        //     .position(|portal| next.0 == portal.0 && next.1 == portal.1)
        //     .unwrap_or(99);
        // if index != 99 {
        //     next = (conf::PORTALS_DEST[index].0, conf::PORTALS_DEST[index].1);
        // }
        if pass_wall <= 0 {
            if banned_points.contains(&next) {
                continue;
            }
        }

        for enemy in enemies_position.clone() {
            // println!("next: {:?}, start: {:?}, enemy: {:?}", next, start, enemy);
            let new_path = algo::a_star_search(next, enemy).unwrap();
            let old_path = algo::a_star_search(start, enemy).unwrap();
            // println!("new_path: {:?}, old_path: {:?}", new_path, old_path);
            if new_path.len() <= old_path.len() {
                all_away = false;
            }
        }
        if all_away {
            return Ok(direction.to_string());
        }
    }

    if enemies_position.len() == 1 && distance(start, enemies_position[0]) == 2 {
        return Ok("STAY".to_string());
    }

    let mut is_diag = true;
    for enemy in enemies_position.clone() {
        if (enemy.0 - start.0).abs() != 1 || (enemy.1 - start.1).abs() != 1 {
            is_diag = false;
            break;
        }
    }
    if is_diag {
        return Ok("STAY".to_string());
    }

    // TODO: stay?
    Ok("STAY".to_string())
}

// BASELINE 170 round end the game
#[pyfunction]
fn collect_coins(
    mut start: Point,
    mut eaten_coins: HashSet<Point>,
) -> PyResult<(Vec<Point>, usize)> {
    let mut agent_coins_score = 0;
    let mut total_path = Vec::new();
    let mut search_coins = 0;
    let depth = 9;
    loop {
        // Pre-calculate ememy's next position
        search_coins += 1;
        let coins: Vec<Point> = conf::COINS
            .par_iter()
            .filter(|x| !eaten_coins.contains(&x))
            .map(|x| (x.0, x.1))
            .collect();

        if coins.is_empty() || search_coins > depth {
            break;
        }

        let paths: Vec<Vec<Point>> = coins
            .iter()
            .filter_map(|&coin| algo::a_star_search(start, coin))
            .collect();

        if paths.is_empty() {
            return Ok((vec![], 0));
        }

        let sp = paths.iter().min_by_key(|path| path.len()).unwrap();
        total_path.extend_from_slice(&sp[..sp.len()]);

        start = *sp.last().unwrap();
        eaten_coins.insert((start.0, start.1));
        agent_coins_score += 2;
    }

    Ok((total_path, agent_coins_score))
}

// BASELINE of defenders with some simple strategies
#[pyfunction]
fn catch_enemies_using_powerup(
    agent_id: usize,
    start: Point,
    pass_wall: usize,
    mut enemies: Vec<Point>,
) -> PyResult<Vec<Point>> {
    let origin = start.clone();

    // if enemies.is_empty() {
    //     let mut explore_path = Vec::new();

    //     let mut targets: Vec<Point> = Vec::new();
    //     match agent_id {
    //         0 => {
    //             for i in 0..12 {
    //                 for j in 0..12 {
    //                     let target = (i, j);
    //                     if conf::WALLS.contains(&target) || start == target {
    //                         continue;
    //                     }
    //                     targets.push(target);
    //                 }
    //             }
    //         }
    //         1 => {
    //             for i in 0..12 {
    //                 for j in 12..24 {
    //                     let target = (i, j);
    //                     if conf::WALLS.contains(&target) || start == target {
    //                         continue;
    //                     }
    //                     targets.push(target);
    //                 }
    //             }
    //         }
    //         2 => {
    //             for i in 12..24 {
    //                 for j in 12..24 {
    //                     let target = (i, j);
    //                     if conf::WALLS.contains(&target) || start == target {
    //                         continue;
    //                     }
    //                     targets.push(target);
    //                 }
    //             }
    //         }
    //         3 => {
    //             for i in 12..24 {
    //                 for j in 0..12 {
    //                     let target = (i, j);
    //                     if conf::WALLS.contains(&target) || start == target {
    //                         continue;
    //                     }
    //                     targets.push(target);
    //                 }
    //             }
    //         }
    //         _ => {}
    //     }

    //     let mut hull_points = algo::graham_hull(targets);
    //     hull_points.sort_by_key(|&p| distance(start, p));
    //     // println!("start: {:?},targets: {:?}", start, targets);
    //     let target_path = algo::a_star_search(start, hull_points[0]).unwrap();

    //     if !target_path.is_empty() {
    //         explore_path.extend_from_slice(&target_path);
    //     }

    //     return Ok(explore_path);
    // } else {
    //     let mut chase_path = Vec::new();
    //     enemies.sort_by_key(|&e| distance(origin, e));
    //     if enemies[0] == start {
    //         for &(i, j) in &[(-1, 0), (1, 0), (0, 1), (0, -1)] {
    //             let new_hp = (enemies[0].0 + i, enemies[0].1 + j);
    //             if !conf::WALLS.contains(&new_hp) {
    //                 chase_path.push(new_hp);
    //                 break;
    //             }
    //         }
    //     } else {
    //         chase_path =
    //             algo::a_star_search_power(origin, enemies[0], vec![], pass_wall).unwrap_or(vec![]);
    //     }
    //     return Ok(chase_path);
    // }

    let mut chase_path = Vec::new();
    enemies.sort_by_key(|&e| distance(origin, e));
    if enemies[0] == start {
        for &(i, j) in &[(-1, 0), (1, 0), (0, 1), (0, -1)] {
            let new_hp = (enemies[0].0 + i, enemies[0].1 + j);
            if !conf::WALLS.contains(&new_hp) {
                chase_path.push(new_hp);
                break;
            }
        }
    } else {
        chase_path =
            algo::a_star_search_power(origin, enemies[0], vec![], pass_wall).unwrap_or(vec![]);
    }
    return Ok(chase_path);
}

#[pyfunction]
fn collect_coins_using_hull(start: Point, eaten_coins: HashSet<Point>) -> PyResult<Vec<Point>> {
    let coins_vec: Vec<Point> = conf::COINS
        .par_iter()
        .map(|x| (x.0, x.1))
        .filter(|p| !eaten_coins.contains(p))
        .collect();

    let mut hull_points = algo::graham_hull(coins_vec);
    hull_points.sort_by_key(|&p| distance(start, p));
    let mut escape_path = Vec::new();

    let mut target_hull_point = start.clone();
    let mut target_dist = i32::MAX;
    for &hp in hull_points.iter() {
        let mut dist = distance(start, hp);

        let mut use_nearby = false;
        let mut new_hp = (hp.0, hp.1);
        if conf::WALLS.contains(&hp) {
            for &(i, j) in &[(-1, 0), (1, 0), (0, 1), (0, -1)] {
                new_hp = (hp.0 + i, hp.1 + j);
                if !conf::WALLS.contains(&new_hp) {
                    use_nearby = true;
                    break;
                }
            }
        }
        if use_nearby {
            dist = distance(start, new_hp);
        }
        if dist < target_dist {
            target_dist = dist;
            if !use_nearby {
                target_hull_point = hp.clone();
            } else {
                target_hull_point = new_hp.clone();
            }
        }
    }
    let hull_path = algo::a_star_search(start, target_hull_point).unwrap();
    escape_path.extend_from_slice(&hull_path);
    Ok(escape_path)
}

// BASELINE of defenders with some simple strategies
#[pyfunction]
fn collect_coins_using_powerup(
    mut start: Point,
    mut eaten_coins: HashSet<Point>,
    allies_position: Vec<Point>,
    enemies_position: Vec<Point>,
    mut pass_wall: usize,
) -> PyResult<(Vec<Point>, usize)> {
    let origin = start.clone();

    let mut search_depth = 0;
    let mut agent_coins_score = 0;
    let mut total_path = Vec::new();

    // check escape strategy
    let escape_path = deal_with_enemy_nearby(start, enemies_position.clone());
    if !escape_path.is_empty() {
        return Ok((escape_path, 0));
    }

    // find the potential move with greatest coin score
    loop {
        search_depth += 1;
        let mut positive_targets: Vec<Point> = conf::COINS
            .par_iter()
            .chain(conf::POWERUPS.par_iter())
            .filter(|x| !eaten_coins.contains(&x))
            .map(|x| (x.0, x.1))
            .collect();
        positive_targets.sort_by_key(|&t| distance(start, t));

        if search_depth > 10 {
            break;
        }

        let mut paths: Vec<Vec<Point>> = positive_targets
            .par_iter()
            .filter_map(|&target| algo::a_star_search_power(start, target, vec![], pass_wall))
            .collect();

        // TODO: away from potential enemies
        if paths.is_empty() {
            // println!("NO TARGET! current position: {:?}", start);
            paths = conf::DEFENDER_BASE
                .par_iter()
                .chain(conf::ATTACKER_BASE.par_iter())
                .filter_map(|&p| algo::a_star_search_power(start, p, vec![], pass_wall))
                .collect();
        }

        if paths.is_empty() {
            println!(
                "No strategy for now: JUST STAY. enemies: {:?}",
                enemies_position
            );
            total_path.push(origin);
            break;
        }

        let sp = paths
            .iter()
            .filter(|path| path.len() > 0)
            .min_by_key(|path| path.len())
            .unwrap();

        total_path.extend_from_slice(&sp[..sp.len()]);

        start = *sp.last().unwrap();
        eaten_coins.insert((start.0, start.1));
        agent_coins_score += 2;
        pass_wall -= sp.len();
    }

    if total_path.len() == 0 {
        println!("no total path generated.")
    }

    Ok((total_path, agent_coins_score))
}

// PERF: all agents parallel
#[pyfunction]
fn explore_n_round_scores(
    start: Point,
    real_eaten_coins: HashSet<Point>,
    enemies_position: Vec<Point>,
    pass_wall: usize,
) -> PyResult<Vec<f32>> {
    let mut banned_points: HashSet<_> = conf::WALLS.iter().cloned().collect();
    banned_points.insert((23, 0));

    // STAY, LEFT, RIGHT, DOWN, UP
    // let mut action_scores = vec![0.0, 0.0, 0.0, 0.0, 0.0];

    let eaten_coins = real_eaten_coins.clone();

    // let visited = HashSet::new();

    let search_depth = match enemies_position.len() {
        0 => 9,
        _ => 3,
    };

    let action_scores = bfs(
        start,
        search_depth,
        pass_wall,
        start,
        vec![],
        &eaten_coins,
        &enemies_position,
        &banned_points,
    );

    Ok(action_scores)
}

#[pyfunction]
fn explore(
    agents: Vec<usize>,
    positions: Vec<HashMap<usize, Point>>,
    map_score: Vec<Vec<f32>>,
    visited: HashMap<usize, HashSet<Point>>,
    max_step: usize,
) -> Vec<f32> {
    let result = Arc::new(Mutex::new(Vec::new()));
    for _ in positions.iter() {
        let mut result = result.lock().unwrap();
        result.push(0.0);
    }
    println!("agents: {:?} pos_len: {}", agents, positions.len());
    positions.par_iter().enumerate().for_each(|(idx, pos)| {
        let mut max_reward = vec![0.0];

        algo::dfs(
            &agents,
            &pos,
            &map_score,
            0.0,
            &visited,
            0,
            max_step,
            &mut max_reward,
        );
        let mut result = result.lock().unwrap();
        result[idx] = max_reward[0];
    });
    Arc::try_unwrap(result).unwrap().into_inner().unwrap()
}

#[pymodule]
fn rust_perf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_direction, m)?)?;
    m.add_function(wrap_pyfunction!(get_direction_path, m)?)?;
    m.add_function(wrap_pyfunction!(collect_coins, m)?)?;
    m.add_function(wrap_pyfunction!(collect_coins_using_powerup, m)?)?;
    m.add_function(wrap_pyfunction!(catch_enemies_using_powerup, m)?)?;
    m.add_function(wrap_pyfunction!(check_stay_or_not, m)?)?;
    m.add_function(wrap_pyfunction!(explore_n_round_scores, m)?)?;
    m.add_function(wrap_pyfunction!(collect_coins_using_hull, m)?)?;
    m.add_function(wrap_pyfunction!(explore, m)?)?;
    Ok(())
}
