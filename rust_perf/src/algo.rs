/**
 * File              : algo.rs
 * Author            : Yue Peng <yuepaang@gmail.com>
 * Date              : 21.09.2023
 * Last Modified Date: 26.09.2023
 * Last Modified By  : Yue Peng <yuepaang@gmail.com>
 */
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use rayon::prelude::*;

use crate::conf;
use crate::distance;
use crate::Direction;
use crate::Node;
use crate::Point;

pub fn astar(start: Point, end: Point, blocked: HashSet<Point>) -> Option<String> {
    let mut banned_points: HashSet<_> = conf::WALLS.iter().cloned().collect();
    banned_points.extend(blocked);

    let mut to_visit = BinaryHeap::new();
    to_visit.push(Node {
        cost: 0,
        point: start,
        passwall: 0,
    });

    let mut visited = HashSet::new();
    let mut g_scores: HashMap<Point, i32> = HashMap::new();
    g_scores.insert(start, 0);

    let mut came_from = HashMap::new();

    while let Some(Node {
        cost: _,
        point,
        passwall: _,
    }) = to_visit.pop()
    {
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
            (0, 1, Direction::Down),
            (0, -1, Direction::Up),
        ] {
            let mut next = (point.0 + i, point.1 + j);

            if visited.contains(&next) || banned_points.contains(&next) {
                continue;
            }

            // PORTAL MOVE
            let index = conf::PORTALS
                .iter()
                .position(|portal| next.0 == portal.0 && next.1 == portal.1)
                .unwrap_or(99);
            if index != 99 {
                next = (conf::PORTALS_DEST[index].0, conf::PORTALS_DEST[index].1);
            }

            let tentative_g_score = g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&i32::MAX) {
                came_from.insert(next, (direction, point));
                g_scores.insert(next, tentative_g_score);
                let f_score = tentative_g_score + distance(next, end);
                to_visit.push(Node {
                    cost: f_score,
                    point: next,
                    passwall: 0,
                });
            }
        }

        visited.insert(point);
    }

    None
}

pub fn astar_path(start: Point, end: Point, blocked: HashSet<Point>) -> Option<(i32, Vec<String>)> {
    let mut banned_points: HashSet<_> = conf::WALLS.iter().cloned().collect();
    banned_points.extend(blocked);

    let mut to_visit = BinaryHeap::new();
    to_visit.push(Node {
        cost: 0,
        point: start,
        passwall: 0,
    });

    let mut visited = HashSet::new();
    let mut g_scores = HashMap::new();
    g_scores.insert(start, 0);

    let mut came_from = HashMap::new();

    while let Some(Node {
        cost,
        point,
        passwall: _,
    }) = to_visit.pop()
    {
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
            (0, 1, Direction::Down),
            (0, -1, Direction::Up),
        ] {
            let mut next = (point.0 + i, point.1 + j);

            if visited.contains(&next) || banned_points.contains(&next) {
                continue;
            }

            // PORTAL MOVE
            let index = conf::PORTALS
                .iter()
                .position(|portal| next.0 == portal.0 && next.1 == portal.1)
                .unwrap_or(99);
            if index != 99 {
                next = (conf::PORTALS_DEST[index].0, conf::PORTALS_DEST[index].1);
            }

            let tentative_g_score = g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&i32::MAX) {
                came_from.insert(next, (direction, point));
                g_scores.insert(next, tentative_g_score);
                let f_score = tentative_g_score + distance(next, end);
                to_visit.push(Node {
                    cost: f_score,
                    point: next,
                    passwall: 0,
                });
            }
        }

        visited.insert(point);
    }

    None
}

pub fn a_star_search(start: Point, end: Point) -> Option<Vec<Point>> {
    let mut banned_points: HashSet<_> = conf::WALLS.iter().cloned().collect();
    banned_points.remove(&end);

    let mut to_visit = BinaryHeap::new();
    to_visit.push(Node {
        cost: 0,
        point: start,
        passwall: 0,
    });

    let mut visited = HashSet::new();
    let mut g_scores = HashMap::new();
    g_scores.insert(start, 0);

    let mut came_from: HashMap<Point, (Direction, Point)> = HashMap::new();

    while let Some(Node {
        cost: _,
        point,
        passwall: _,
    }) = to_visit.pop()
    {
        if point == end {
            let mut path: Vec<Point> = Vec::new();
            let mut current = end;

            while let Some(&(direction, parent)) = came_from.get(&current) {
                let next_pos: Point = direction.get_next_pos(parent);
                path.push(next_pos);
                current = parent;
            }

            path.reverse();
            return Some(path);
        }

        for &(i, j, direction) in &[
            (0, 0, Direction::Stay),
            (-1, 0, Direction::Left),
            (1, 0, Direction::Right),
            (0, 1, Direction::Down),
            (0, -1, Direction::Up),
        ] {
            let mut next = (point.0 + i, point.1 + j);
            let true_next = next.clone();
            if check_out_of_bound(next) {
                continue;
            }
            if visited.contains(&next) || banned_points.contains(&next) {
                continue;
            }

            // PORTAL MOVE
            let index = conf::PORTALS
                .iter()
                .position(|portal| next.0 == portal.0 && next.1 == portal.1)
                .unwrap_or(99);
            if index != 99 {
                next = (conf::PORTALS_DEST[index].0, conf::PORTALS_DEST[index].1);
            }

            let tentative_g_score = g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&i32::MAX) {
                came_from.insert(next, (direction, point));
                g_scores.insert(next, tentative_g_score);
                let f_score = tentative_g_score + crate::distance(next, end);
                to_visit.push(Node {
                    cost: f_score,
                    point: next,
                    passwall: 0,
                });
            }

            if true_next == end {
                came_from.insert(true_next, (direction, point));
                let mut path: Vec<Point> = Vec::new();
                let mut current = end;

                while let Some(&(direction, parent)) = came_from.get(&current) {
                    let next_pos: Point = direction.get_next_pos(parent);
                    path.push(next_pos);
                    current = parent;
                }

                path.reverse();
                return Some(path);
            }
        }

        visited.insert(point);
    }
    None
}

pub fn move_enemy(enemy_position: Point, target_position: Point) -> Point {
    if let Some(path) = a_star_search(enemy_position, target_position) {
        if path.len() >= 1 {
            return path[0];
        }
    }
    enemy_position.clone()
}

pub fn a_star_search_power(
    start: Point,
    end: Point,
    allies: Vec<Point>,
    passwall: usize,
) -> Option<Vec<Point>> {
    let mut banned_points: HashSet<_> = conf::WALLS.iter().cloned().collect();
    // valueless position
    banned_points.insert((23, 0));
    // TODO: try
    for p in conf::PORTALS.iter() {
        banned_points.insert(*p);
    }
    for p in conf::PORTALS_DEST.iter() {
        banned_points.insert(*p);
    }

    let mut to_visit = BinaryHeap::new();
    to_visit.push(Node {
        cost: 0,
        point: start,
        passwall,
    });

    let mut visited = HashSet::new();
    let mut g_scores = HashMap::new();
    g_scores.insert(start, 0);

    let mut came_from: HashMap<Point, (Direction, Point)> = HashMap::new();

    while let Some(Node {
        cost: _,
        point,
        passwall: pw,
    }) = to_visit.pop()
    {
        if point == end {
            let mut path: Vec<Point> = Vec::new();
            let mut current = end;

            while let Some(&(direction, parent)) = came_from.get(&current) {
                let next_pos: Point = direction.get_next_pos(parent);
                path.push(next_pos);
                current = parent;
            }

            path.reverse();
            return Some(path);
        }

        for &(i, j, direction) in &[
            (0, 0, Direction::Stay),
            (-1, 0, Direction::Left),
            (1, 0, Direction::Right),
            (0, 1, Direction::Down),
            (0, -1, Direction::Up),
        ] {
            let mut next = (point.0 + i, point.1 + j);
            if check_out_of_bound(next) {
                continue;
            }
            if visited.contains(&next) {
                continue;
            }
            if allies.contains(&next) {
                continue;
            }
            if pw <= 0 {
                if banned_points.contains(&next) {
                    continue;
                }
            }

            // PORTAL MOVE
            let index = conf::PORTALS
                .iter()
                .position(|portal| next.0 == portal.0 && next.1 == portal.1)
                .unwrap_or(99);
            if index != 99 {
                next = (conf::PORTALS_DEST[index].0, conf::PORTALS_DEST[index].1);
            }

            let tentative_g_score = g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&i32::MAX) {
                came_from.insert(next, (direction, point));
                g_scores.insert(next, tentative_g_score);
                let f_score = tentative_g_score + crate::distance(next, end);
                to_visit.push(Node {
                    cost: f_score,
                    point: next,
                    passwall: pw - 1,
                });
            }
        }

        visited.insert(point);
    }
    None
}

pub fn check_out_of_bound(point: Point) -> bool {
    return point.0 < 0 || point.0 >= conf::WIDTH || point.1 < 0 || point.1 >= conf::HEIGHT;
}

pub fn deal_with_enemy_nearby(start: Point, enemies: Vec<Point>) -> Vec<Point> {
    let mut escape_path = Vec::new();
    if enemies.is_empty() {
        return escape_path;
    }
    let enemies_dist: Vec<i32> = enemies.iter().map(|&e| distance(start, e)).collect();
    let min_dist = enemies_dist.iter().min().unwrap();

    for &(i, j) in &[(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)] {
        if min_dist == &1 {
            if i == 0 && j == 0 {
                continue;
            }
        }
        let mut next = (start.0 + i, start.1 + j);
        if check_out_of_bound(next) {
            continue;
        }

        let true_next = next.clone();
        // PORTAL MOVE
        let index = conf::PORTALS
            .iter()
            .position(|portal| next.0 == portal.0 && next.1 == portal.1)
            .unwrap_or(99);
        if index != 99 {
            next = (conf::PORTALS_DEST[index].0, conf::PORTALS_DEST[index].1);
        }

        let mut flag = false;
        conf::WALLS.iter().for_each(|&w| {
            if w == next {
                if !flag {
                    flag = true;
                }
            }
        });
        // skip direction into wall
        if flag {
            continue;
        }

        flag = false;
        enemies.iter().for_each(|&e| {
            if distance(next, e) > distance(start, e) {
                if !flag {
                    flag = true;
                }
            }
        });
        // direction away from enemy
        if flag {
            escape_path.push(true_next);
        }
    }

    // if no simple escape path
    if escape_path.is_empty() {
        let walls_vec: Vec<Point> = conf::WALLS
            .iter()
            .chain(enemies.iter())
            .map(|x| (x.0, x.1))
            .filter(|&p| distance(start, p) <= 5)
            .collect();

        let hull_points = graham_hull(walls_vec);

        let mut target_hull_point = start.clone();
        let mut target_dist = 0;
        for &hp in hull_points.iter() {
            let mut dist = 0;

            let mut use_nearby = false;
            let mut new_hp = (hp.0, hp.1);
            if !conf::WALLS.contains(&hp) {
                for &e in enemies.iter() {
                    dist += distance(hp, e);
                }
            } else {
                for &(i, j) in &[(-1, 0), (1, 0), (0, 1), (0, -1)] {
                    new_hp = (hp.0 + i, hp.1 + j);
                    if !conf::WALLS.contains(&new_hp) {
                        for &e in enemies.iter() {
                            dist += distance(new_hp, e);
                        }
                        use_nearby = true;
                        break;
                    }
                }
            }
            if dist > target_dist {
                target_dist = dist;
                if !use_nearby {
                    target_hull_point = hp.clone();
                } else {
                    target_hull_point = new_hp.clone();
                }
            }
        }
        let hull_path = a_star_search(start, target_hull_point).unwrap();
        println!(
            "EMPTY ESCAPE!!!start{:?}, enemies: {:?}, hull_path: {:?}",
            start, enemies, hull_path
        );
        escape_path.extend_from_slice(&hull_path);
    }

    escape_path
}

fn orientation(p: Point, q: Point, r: Point) -> i32 {
    let val = (q.1 - p.1) * (r.0 - q.0) - (q.0 - p.0) * (r.1 - q.1);
    if val == 0 {
        0
    } else if val > 0 {
        1
    } else {
        -1
    }
}

pub fn graham_hull(points: Vec<Point>) -> Vec<Point> {
    let mut points = points.clone();
    points.sort_by(|p, q| p.0.cmp(&q.0).then(p.1.cmp(&q.1)));

    let mut lower = Vec::new();
    for &p in points.iter() {
        while lower.len() >= 2
            && orientation(lower[lower.len() - 2], *lower.last().unwrap(), p) != -1
        {
            lower.pop();
        }
        lower.push(p);
    }

    let mut upper = Vec::new();
    for &p in points.iter().rev() {
        while upper.len() >= 2
            && orientation(upper[upper.len() - 2], *upper.last().unwrap(), p) != -1
        {
            upper.pop();
        }
        upper.push(p);
    }

    upper.pop();
    lower.pop();
    lower.extend_from_slice(&upper);
    lower
}

pub fn dfs(
    root: Point,
    search_depth: usize,
    pass_wall: usize,
    current_point: Point,
    path: Vec<Point>,
    eaten_coins: &mut HashSet<Point>,
    enemies: &Vec<Point>,
    banned_points: &HashSet<Point>,
    action_score: &mut Vec<f32>,
    mut first_move_flag: i8,
) {
    if search_depth > 9 {
        return;
    }
    // Pre-calculate ememys' next position (assume chasing if in their vision)
    let next_enemies = enemies
        .iter()
        .map(|&enemy| move_enemy(enemy.clone(), current_point))
        .collect::<Vec<Point>>();

    for &(action_idx, i, j) in &[(0, 0, 0), (1, -1, 0), (2, 1, 0), (3, 0, 1), (4, 0, -1)] {
        if search_depth == 0 {
            first_move_flag = action_idx;
        }

        if enemies.is_empty() {
            if action_idx == 0 {
                continue;
            }
        }

        let mut next = (current_point.0 + i, current_point.1 + j);
        if check_out_of_bound(next) || path.contains(&next) {
            continue;
        }
        // PORTAL MOVE
        let portail_index = conf::PORTALS
            .par_iter()
            .position_first(|portal| next.0 == portal.0 && next.1 == portal.1)
            .unwrap_or(99);
        if portail_index != 99 {
            next = (
                conf::PORTALS_DEST[portail_index].0,
                conf::PORTALS_DEST[portail_index].1,
            );
        }

        if pass_wall <= 0 {
            if banned_points.contains(&next) {
                continue;
            }
        }

        // TODO: score calculation
        if conf::COINS.contains(&next) && !eaten_coins.contains(&next) {
            action_score[first_move_flag as usize] += f32::powf(0.95, search_depth as f32) * 2.0;
            // println!("next: {:?}, path: {:?}", next, path);
            eaten_coins.insert(next);
        }
        if next_enemies.contains(&next) {
            action_score[first_move_flag as usize] /= 2.0;
        }

        let mut new_path = path.clone();
        new_path.push(next);
        dfs(
            root,
            search_depth + 1,
            pass_wall - 1,
            next,
            new_path,
            eaten_coins,
            &next_enemies,
            banned_points,
            action_score,
            first_move_flag,
        );
        if search_depth == 0 {
            eaten_coins.remove(&next);
        }
    }
}

pub fn bfs(
    root: Point,
    search_depth: usize,
    pass_wall: usize,
    current_point: Point,
    path: Vec<Point>,
    eaten_coins: &HashSet<Point>,
    enemies: &Vec<Point>,
    banned_points: &HashSet<Point>,
) -> Vec<f32> {
    // STAY, LEFT, RIGHT, DOWN, UP
    let mut action_scores = vec![0.0, 0.0, 0.0, 0.0, 0.0];

    let mut queue: VecDeque<_> = VecDeque::new();
    queue.push_back((
        root,
        1,
        pass_wall,
        current_point,
        path.clone(),
        0,
        eaten_coins.clone(),
    ));

    while let Some((
        current_root,
        current_depth,
        current_pass_wall,
        current_point,
        current_path,
        mut current_first_move_flag,
        virtual_coins,
    )) = queue.pop_front()
    {
        if current_depth > search_depth {
            continue;
        }

        let next_enemies = enemies
            .iter()
            .map(|&enemy| move_enemy(enemy.clone(), current_point))
            .collect::<Vec<Point>>();

        for &(action_idx, i, j) in &[(0, 0, 0), (1, -1, 0), (2, 1, 0), (3, 0, 1), (4, 0, -1)] {
            if current_depth == 1 {
                current_first_move_flag = action_idx;
            }

            if enemies.is_empty() && action_idx == 0 {
                continue;
            }

            let mut next = (current_point.0 + i, current_point.1 + j);
            if check_out_of_bound(next) {
                continue;
            }

            if current_path.contains(&next) {
                continue;
            }

            // PORTAL MOVE
            let portail_index = conf::PORTALS
                .par_iter()
                .position_first(|portal| next.0 == portal.0 && next.1 == portal.1)
                .unwrap_or(99);
            if portail_index != 99 {
                next = (
                    conf::PORTALS_DEST[portail_index].0,
                    conf::PORTALS_DEST[portail_index].1,
                );
            }

            // if enemies.is_empty() {
            //     continue;
            // }

            if pass_wall <= 0 && banned_points.contains(&next) {
                continue;
            }
            let mut cloned_coins = virtual_coins.clone();
            // TODO: score calculation
            if conf::COINS.contains(&next) && !cloned_coins.contains(&next) {
                action_scores[current_first_move_flag as usize] +=
                    f32::powf(0.95, current_depth as f32) * 2.0;
                cloned_coins.insert(next);
            }
            // for c in conf::COINS.iter() {
            //     if !eaten_coins.contains(c) {
            //         let dist = distance(next, *c);
            //         if dist == 0 {
            //             action_scores[current_first_move_flag as usize] +=
            //                 1000.0 / current_depth as f32
            //         } else {
            //             action_scores[current_first_move_flag as usize] +=
            //                 f32::powf(0.95, current_depth as f32) * (1.0 / dist as f32) * 2.0;
            //         }
            //     }
            // }

            if next_enemies.contains(&next) {
                action_scores[current_first_move_flag as usize] /= 2.0;
            }

            let mut new_path = current_path.clone();
            new_path.push(next);
            queue.push_back((
                current_root,
                current_depth + 1,
                current_pass_wall - 1,
                next,
                new_path,
                current_first_move_flag,
                cloned_coins,
            ));
        }
    }
    action_scores
}
