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
    banned_points.remove(&end);

    let mut portals = HashMap::new();
    for (idx, portal) in conf::PORTALS.iter().enumerate() {
        portals.insert(portal, conf::PORTALS_DEST[idx].clone());
    }

    let mut to_visit = BinaryHeap::new();
    to_visit.push(Node {
        cost: 0,
        point: start,
        passwall: 0,
        shield: 0,
        depth: 0,
    });

    let mut visited = HashSet::new();
    let mut g_scores: HashMap<Point, i32> = HashMap::new();
    g_scores.insert(start, 0);

    let mut came_from = HashMap::new();

    while let Some(Node {
        cost: _,
        point,
        passwall: _,
        shield: _,
        depth: _,
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

        // Check if current point is a portal entrance
        if let Some(&next) = portals.get(&point) {
            let tentative_g_score = g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&i32::MAX) {
                came_from.insert(next, (Direction::Stay, point));
                g_scores.insert(next, tentative_g_score);
                let f_score = tentative_g_score;
                to_visit.push(Node {
                    cost: f_score,
                    point: next,
                    passwall: 0,
                    shield: 0,
                    depth: 0,
                });
            }
        }

        for &(i, j, direction) in &[
            (-1, 0, Direction::Left),
            (1, 0, Direction::Right),
            (0, 1, Direction::Down),
            (0, -1, Direction::Up),
        ] {
            let next = (point.0 + i, point.1 + j);

            if check_out_of_bound(next) {
                continue;
            }

            if visited.contains(&next) || banned_points.contains(&next) {
                continue;
            }

            let tentative_g_score = g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&i32::MAX) {
                came_from.insert(next, (direction, point));
                g_scores.insert(next, tentative_g_score);
                let f_score = tentative_g_score + distance(next, end);
                to_visit.push(Node {
                    cost: f_score,
                    point: next,
                    shield: 0,
                    passwall: 0,
                    depth: 0,
                });
            }
        }

        visited.insert(point);
    }

    None
}

pub fn astar_path(start: Point, end: Point, blocked: HashSet<Point>) -> Option<(i32, Vec<String>)> {
    let mut banned_points: HashSet<_> = conf::WALLS.iter().cloned().collect();
    banned_points.extend(blocked.clone());
    banned_points.remove(&end);
    // println!(
    //     "start: {:?}, end: {:?}, blocked :{:?}",
    //     start,
    //     end,
    //     blocked.clone()
    // );

    let mut portals = HashMap::new();
    for (idx, portal) in conf::PORTALS.iter().enumerate() {
        portals.insert(portal, conf::PORTALS_DEST[idx].clone());
    }

    let mut to_visit = BinaryHeap::new();
    to_visit.push(Node {
        cost: 0,
        point: start,
        passwall: 0,
        shield: 0,
        depth: 0,
    });

    let mut visited = HashSet::new();
    let mut g_scores = HashMap::new();
    g_scores.insert(start, 0);

    let mut came_from = HashMap::new();

    while let Some(Node {
        cost,
        point,
        passwall: _,
        shield: _,
        depth: _,
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

        // Check if current point is a portal entrance
        if let Some(&next) = portals.get(&point) {
            let tentative_g_score = g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&i32::MAX) {
                came_from.insert(next, (Direction::Stay, point));
                g_scores.insert(next, tentative_g_score);
                let f_score = tentative_g_score;
                to_visit.push(Node {
                    cost: f_score,
                    point: next,
                    passwall: 0,
                    shield: 0,
                    depth: 0,
                });
            }
        }

        for &(i, j, direction) in &[
            (-1, 0, Direction::Left),
            (1, 0, Direction::Right),
            (0, 1, Direction::Down),
            (0, -1, Direction::Up),
        ] {
            let next = (point.0 + i, point.1 + j);

            if check_out_of_bound(next) {
                continue;
            }

            if visited.contains(&next) || banned_points.contains(&next) {
                continue;
            }

            let tentative_g_score = g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&i32::MAX) {
                came_from.insert(next, (direction, point));
                g_scores.insert(next, tentative_g_score);
                let f_score = tentative_g_score;
                to_visit.push(Node {
                    cost: f_score,
                    point: next,
                    passwall: 0,
                    shield: 0,
                    depth: 0,
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
        shield: 0,
        depth: 0,
    });

    let mut visited = HashSet::new();
    let mut g_scores = HashMap::new();
    g_scores.insert(start, 0);

    let mut came_from: HashMap<Point, (Direction, Point)> = HashMap::new();

    while let Some(Node {
        cost: _,
        point,
        passwall: _,
        shield: _,
        depth: _,
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
                    shield: 0,
                    depth: 0,
                });
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

fn enemy_penalty(point: Point, enemies: &Vec<Point>, penalty_range: i32) -> i32 {
    let mut penalty = 0;
    for &enemy in enemies {
        let dist = crate::distance(point, enemy);
        if dist <= penalty_range {
            penalty += penalty_range - dist; // You can adjust this formula as needed
        }
    }
    penalty as i32
}

pub fn a_star_search_power(
    start: Point,
    end: Point,
    passwall: usize,
    shield: usize,
    init_banned_points: &HashSet<Point>,
    enemies_all_pos: &HashSet<Point>,
    enemies_next_all_pos: &HashSet<Point>,
    openness_map: &HashMap<Point, i32>,
) -> Option<Vec<Point>> {
    let mut portals = HashMap::new();
    for (idx, portal) in conf::PORTALS.iter().enumerate() {
        portals.insert(portal, conf::PORTALS_DEST[idx].clone());
    }
    let mut banned_points = init_banned_points.clone();
    banned_points.remove(&end);

    let enemies: Vec<Point> = enemies_all_pos.par_iter().map(|x| (x.0, x.1)).collect();

    let mut to_visit = BinaryHeap::new();
    to_visit.push(Node {
        cost: 0,
        point: start,
        passwall,
        shield,
        depth: 0,
    });

    let mut visited = HashSet::new();
    let mut g_scores = HashMap::new();
    g_scores.insert(start, 0);

    let mut came_from: HashMap<Point, (Direction, Point)> = HashMap::new();

    while let Some(Node {
        cost: _,
        point,
        passwall: pw,
        shield: sh,
        depth: dep,
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

        // Check if current point is a portal entrance
        if let Some(&next) = portals.get(&point) {
            let tentative_g_score = g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&i32::MAX) {
                came_from.insert(next, (Direction::Stay, point));
                g_scores.insert(next, tentative_g_score);
                let mut openness = 0;
                openness += openness_map.get(&next).unwrap_or(&1) - 1;
                let mut penalty = 0;
                // penalty += enemy_penalty(next, &enemies, 3.0);
                let f_score = tentative_g_score - openness + penalty;
                if !enemies_all_pos.contains(&next) && !banned_points.contains(&next) {
                    to_visit.push(Node {
                        cost: f_score,
                        point: next,
                        passwall: pw,
                        shield: sh,
                        depth: dep,
                    });
                }
            }
        }

        for &(i, j, direction) in &[
            (0, 0, Direction::Stay),
            (-1, 0, Direction::Left),
            (1, 0, Direction::Right),
            (0, 1, Direction::Down),
            (0, -1, Direction::Up),
        ] {
            let next = (point.0 + i, point.1 + j);
            if check_out_of_bound(next) {
                continue;
            }
            if visited.contains(&next) {
                continue;
            }
            if pw == 0 {
                if banned_points.contains(&next) {
                    continue;
                }
            }
            if sh == 0 && dep == 0 {
                if enemies_all_pos.contains(&next) {
                    continue;
                }
            }
            if sh == 0 && dep == 1 {
                if enemies_next_all_pos.contains(&next) {
                    continue;
                }
            }

            let tentative_g_score = *g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&i32::MAX) {
                came_from.insert(next, (direction, point));
                g_scores.insert(next, tentative_g_score);
                let mut openness = 0;
                let mut penalty = 0;
                openness += openness_map.get(&next).unwrap_or(&1) - 1;
                // penalty += enemy_penalty(next, &enemies, 6);
                let f_score = tentative_g_score - openness + penalty;
                let mut new_pw = 0;
                if pw > 0 {
                    new_pw = pw - 1;
                }
                let mut new_sh = 0;
                if sh > 0 {
                    new_sh = sh - 1;
                }

                to_visit.push(Node {
                    cost: f_score,
                    point: next,
                    passwall: new_pw,
                    shield: new_sh,
                    depth: dep + 1,
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

pub fn get_all_nearby_pos(
    agents: &Vec<usize>,
    agent_pos: &HashMap<usize, Point>,
    visited: &HashMap<usize, HashSet<Point>>,
) -> Vec<HashMap<usize, Point>> {
    let delta = [(0, 1), (0, -1), (1, 0), (-1, 0)];

    let mut new_pos: HashMap<usize, Vec<Point>> = HashMap::new();
    for (&id, _) in agent_pos.iter() {
        let mut positions = Vec::new();
        for &d in &delta {
            let new_position = (
                (agent_pos[&id].0 as isize + d.0) as i32,
                (agent_pos[&id].1 as isize + d.1) as i32,
            );
            if check_out_of_bound(new_position) {
                continue;
            }
            if conf::WALLS.contains(&new_position) {
                continue;
            }
            if !visited[&id].contains(&new_position) {
                positions.push(new_position);
            }
        }
        new_pos.insert(id, positions);
    }

    // This will need adjustment to work similarly to the nested loops in the Python version
    let mut next_pos = Vec::new();
    // Fill in next_pos based on new_pos
    for &p0 in &new_pos[&agents[0]] {
        for &p1 in &new_pos[&agents[1]] {
            for &p2 in &new_pos[&agents[2]] {
                for &p3 in &new_pos[&agents[3]] {
                    let mut position_map = HashMap::new();
                    position_map.insert(agents[0], p0);
                    position_map.insert(agents[1], p1);
                    position_map.insert(agents[2], p2);
                    position_map.insert(agents[3], p3);
                    next_pos.push(position_map);
                }
            }
        }
    }

    next_pos
}

fn sp(start: Point, end: Point) -> Option<i32> {
    if start == end {
        return Some(0);
    }
    let directions: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];
    let walls: HashSet<Point> = conf::WALLS.par_iter().map(|x| (x.0, x.1)).collect();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut portals = HashMap::new();
    for (idx, portal) in conf::PORTALS.iter().enumerate() {
        portals.insert(portal, conf::PORTALS_DEST[idx].clone());
    }
    queue.push_back((start, 0));

    while let Some((current, dist)) = queue.pop_front() {
        if current == end {
            return Some(dist);
        }

        // Check if current point is a portal entrance
        if let Some(&portal_exit) = portals.get(&current) {
            if !visited.contains(&portal_exit) {
                visited.insert(portal_exit);
                queue.push_back((portal_exit, dist + 1));
                continue;
            }
        }

        for &direction in directions.iter() {
            let next_point = (current.0 + direction.0, current.1 + direction.1);

            if !visited.contains(&next_point) {
                if walls.contains(&next_point) && next_point != end {
                    continue;
                }

                visited.insert(next_point);
                queue.push_back((next_point, dist + 1));
            }
        }
    }

    None
}

pub fn dfs(
    agents: &Vec<usize>,
    positions: &HashMap<usize, Point>,
    my_pos: &HashMap<usize, Point>,
    mut reward: i32,
    visited: &HashMap<usize, HashSet<Point>>,
    step: usize,
    max_step: usize,
    max_reward: &mut Vec<i32>,
) {
    if step > max_step {
        return;
    }

    let mut visited_copy = HashMap::new();
    for (id, set) in visited.iter() {
        visited_copy.insert(*id, set.clone());
    }

    for (id, &agent_pos) in positions.iter() {
        visited_copy
            .entry(*id)
            .or_insert_with(HashSet::new)
            .insert(agent_pos.clone());
        for (_, &mp) in my_pos.iter() {
            // println!("agent_pos: {:?}, mp: {:?}", agent_pos, mp);
            reward += 24 * 24 - sp(agent_pos, mp).unwrap_or(0);
        }
    }

    if reward > max_reward[0] {
        max_reward[0] = reward;
    }

    let next_position = get_all_nearby_pos(agents, &positions, &visited_copy);

    for p in &next_position {
        dfs(
            agents,
            p,
            my_pos,
            reward,
            &visited_copy,
            step + 1,
            max_step,
            max_reward,
        );
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

pub fn check_valid_move(point: Point) -> usize {
    let mut count = 0;
    for &(i, j) in &[(-1, 0), (1, 0), (0, 1), (0, -1)] {
        let next = (point.0 + i, point.1 + j);
        if !conf::WALLS.contains(&next) {
            count += 1;
        }
    }
    count
}
