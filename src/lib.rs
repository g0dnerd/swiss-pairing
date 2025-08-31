use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    hash::Hash,
};

pub trait Player: Clone + Eq + Hash + Display {}

pub trait Match: Clone {
    type P: Player;

    const POINTS_PER_WIN: u16;
    const POINTS_PER_DRAW: u16;

    fn player1(&self) -> &Self::P;
    /// If player1 has a bye, leave player2 as `None`
    fn player2(&self) -> Option<&Self::P>;

    fn round(&self) -> usize;

    fn player1_wins(&self) -> Option<u8>;
    fn player2_wins(&self) -> Option<u8>;
    fn is_bye(&self) -> bool;
}

pub trait Pairing: Sized {
    type P: Player;

    fn new(player1: Self::P, player2: Option<Self::P>, round: usize) -> Self;
}

#[derive(Default, Clone)]
struct PlayerStats {
    points: u16,
    opponents: HashSet<usize>,
    games_played: usize,
}

struct PairingGraph {
    weights: Vec<Vec<i32>>,
    n: usize,
}

impl PairingGraph {
    fn new(n: usize) -> Self {
        Self {
            weights: vec![vec![0; n]; n],
            n,
        }
    }

    fn set_weight(&mut self, i: usize, j: usize, weight: i32) {
        assert!(i < self.n);
        assert!(j < self.n);

        self.weights[i][j] = weight;
        self.weights[j][i] = weight;
    }

    fn get_weight(&self, i: usize, j: usize) -> i32 {
        assert!(i < self.n);
        assert!(j < self.n);

        self.weights[i][j]
    }

    fn is_valid_pairing(&self, i: usize, j: usize) -> bool {
        assert!(i < self.n);
        assert!(j < self.n);

        self.weights[i][j] > i32::MIN
    }
}

#[derive(Default)]
pub struct SwissPairing<P: Player> {
    _phantom: std::marker::PhantomData<P>,
}

impl<P: Player> SwissPairing<P> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn pair<M, O>(&self, matches: &[M], round: usize) -> Vec<O>
    where
        M: Match<P = P>,
        O: Pairing<P = P>,
    {
        if matches.is_empty() && round == 1 {
            return self.initial_pairings(matches);
        }

        let players = self.collect_players(matches);
        if players.is_empty() {
            return vec![];
        }

        let player_map: HashMap<P, usize> = players
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i))
            .collect();

        let stats = self.calculate_stats(matches, &player_map);
        let graph = self.build_pairing_graph(&players, &stats);

        let matching = self.maximum_weight_matching(&graph, &stats);

        self.create_pairings_from_matching(&matching, &players, round)
    }

    fn initial_pairings<M, O>(&self, _matches: &[M]) -> Vec<O>
    where
        M: Match<P = P>,
        O: Pairing<P = P>,
    {
        todo!()
    }

    fn collect_players<M>(&self, matches: &[M]) -> Vec<P>
    where
        M: Match<P = P>,
    {
        let mut players = HashSet::new();
        for m in matches {
            players.insert(m.player1().clone());
            if let Some(p2) = m.player2() {
                players.insert(p2.clone());
            }
        }

        let mut player_vec: Vec<P> = players.into_iter().collect();
        player_vec.sort_by_key(|p| p.to_string());
        player_vec
    }

    fn calculate_stats<M>(&self, matches: &[M], player_map: &HashMap<P, usize>) -> Vec<PlayerStats>
    where
        M: Match<P = P>,
    {
        let mut stats = vec![PlayerStats::default(); player_map.len()];

        for m in matches {
            let p1_idx = player_map[m.player1()];

            if m.is_bye() {
                stats[p1_idx].points += M::POINTS_PER_WIN;
                stats[p1_idx].games_played += 2;
            } else if let Some(p2) = m.player2() {
                let p2_idx = player_map[p2];
                let p1_wins = m
                    .player1_wins()
                    .expect("Non-bye game doesn't have result data");
                let p2_wins = m
                    .player2_wins()
                    .expect("Non-bye game doesn't have result data");

                let total_games = usize::from(p1_wins + p2_wins);

                stats[p1_idx].games_played += total_games;
                stats[p2_idx].games_played += total_games;
                stats[p1_idx].opponents.insert(p2_idx);
                stats[p2_idx].opponents.insert(p1_idx);

                if p1_wins > p2_wins {
                    stats[p1_idx].points += M::POINTS_PER_WIN;
                } else if p2_wins > p1_wins {
                    stats[p2_idx].points += M::POINTS_PER_WIN;
                } else {
                    stats[p1_idx].points += M::POINTS_PER_DRAW;
                    stats[p2_idx].points += M::POINTS_PER_DRAW;
                }
            }
        }

        stats
    }

    fn build_pairing_graph(&self, players: &[P], stats: &[PlayerStats]) -> PairingGraph {
        let n = players.len();
        let mut graph = PairingGraph::new(n);

        (0..n).for_each(|i| {
            (i + 1..n).for_each(|j| {
                if stats[i].opponents.contains(&j) {
                    graph.set_weight(i, j, i32::MIN);
                } else {
                    // TODO: Tiebreakers
                    let point_diff = stats[i].points.abs_diff(stats[j].points);
                    let weight = i32::from(1000 - point_diff * 100);
                    graph.set_weight(i, j, weight);
                }
            });
        });

        graph
    }

    fn maximum_weight_matching(
        &self,
        graph: &PairingGraph,
        stats: &[PlayerStats],
    ) -> Vec<Option<usize>> {
        let n = graph.n;
        let mut matching = vec![None; n];

        let mut player_order: Vec<usize> = (0..n).collect();
        player_order.sort_by_key(|&i| std::cmp::Reverse(stats[i].points));

        if self.backtrack_matching(graph, &mut matching, &player_order, 0) {
            return matching;
        }

        self.find_best_partial_matching(graph, stats)
    }

    #[allow(clippy::only_used_in_recursion)]
    fn backtrack_matching(
        &self,
        graph: &PairingGraph,
        matching: &mut [Option<usize>],
        player_order: &[usize],
        idx: usize,
    ) -> bool {
        if idx >= player_order.len() {
            return true;
        }

        let player = player_order[idx];

        if matching[player].is_some() {
            return self.backtrack_matching(graph, matching, player_order, idx + 1);
        }

        let mut candidates: Vec<(usize, i32)> = Vec::new();
        (0..matching.len()).for_each(|j| {
            if j != player && matching[j].is_none() && graph.is_valid_pairing(player, j) {
                candidates.push((j, graph.get_weight(player, j)));
            }
        });

        candidates.sort_by_key(|&(_, w)| std::cmp::Reverse(w));

        for (j, _) in candidates {
            matching[player] = Some(j);
            matching[j] = Some(player);

            if self.backtrack_matching(graph, matching, player_order, idx + 1) {
                return true;
            }

            matching[player] = None;
            matching[j] = None;
        }

        false
    }

    fn find_best_partial_matching(
        &self,
        graph: &PairingGraph,
        stats: &[PlayerStats],
    ) -> Vec<Option<usize>> {
        let n = graph.n;
        let mut matching = vec![None; n];
        let mut used = vec![false; n];

        let mut player_order: Vec<usize> = (0..n).collect();
        player_order.sort_by_key(|&i| std::cmp::Reverse(stats[i].points));

        for &i in &player_order {
            if used[i] {
                continue;
            }

            let mut best_j = None;
            let mut best_weight = i32::MIN;

            (0..n).for_each(|j| {
                if i != j && !used[j] && graph.is_valid_pairing(i, j) {
                    let weight = graph.get_weight(i, j);
                    if weight > best_weight {
                        best_weight = weight;
                        best_j = Some(j);
                    }
                }
            });

            if let Some(j) = best_j {
                matching[i] = Some(j);
                matching[j] = Some(i);
                used[i] = true;
                used[j] = true;
            }
        }

        matching
    }

    fn create_pairings_from_matching<O>(
        &self,
        matching: &[Option<usize>],
        players: &[P],
        round: usize,
    ) -> Vec<O>
    where
        O: Pairing<P = P>,
    {
        let mut pairings = Vec::new();
        let mut processed = vec![false; players.len()];

        for i in 0..players.len() {
            if processed[i] {
                continue;
            }

            match matching[i] {
                Some(j) => {
                    pairings.push(O::new(players[i].clone(), Some(players[j].clone()), round));
                    processed[i] = true;
                    processed[j] = true;
                }
                None => {
                    pairings.push(O::new(players[i].clone(), None, round));
                    processed[i] = true;
                }
            }
        }
        pairings
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Display;

    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct SimplePlayer(pub String);
    impl Player for SimplePlayer {}

    impl Display for SimplePlayer {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    #[derive(Clone)]
    pub struct SimpleMatch<P: Player> {
        player1: P,
        player2: Option<P>,
        player1_wins: Option<u8>,
        player2_wins: Option<u8>,

        is_bye: bool,
        pub round: usize,
    }

    impl<P: Player> Display for SimpleMatch<P> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let p1_wins = match self.player1_wins {
                Some(w) => format!("{w}"),
                None => "n/a".to_string(),
            };
            let p2_wins = match self.player2_wins {
                Some(w) => format!("{w}"),
                None => "n/a".to_string(),
            };
            let p2 = match &self.player2 {
                Some(p) => format!("{p}"),
                None => "BYE".to_string(),
            };
            write!(f, "{} vs {}: {} - {}", self.player1, p2, p1_wins, p2_wins)
        }
    }

    impl<P: Player> Match for SimpleMatch<P> {
        type P = P;

        fn player1(&self) -> &Self::P {
            &self.player1
        }
        fn player2(&self) -> Option<&Self::P> {
            self.player2.as_ref()
        }
        fn round(&self) -> usize {
            self.round
        }

        const POINTS_PER_WIN: u16 = 3;
        const POINTS_PER_DRAW: u16 = 1;

        fn player1_wins(&self) -> Option<u8> {
            self.player1_wins
        }

        fn player2_wins(&self) -> Option<u8> {
            self.player2_wins
        }

        fn is_bye(&self) -> bool {
            self.is_bye
        }
    }

    impl<P: Player> Pairing for SimpleMatch<P> {
        type P = P;

        fn new(player1: P, player2: Option<P>, round: usize) -> Self {
            let is_bye = player2.is_none();

            Self {
                player1,
                player2,
                player1_wins: None,
                player2_wins: None,
                is_bye,
                round,
            }
        }
    }

    const TEST_PLAYERS: [&str; 8] = [
        "Anna", "Bella", "Charlie", "Donovan", "Emil", "Fae", "Gulliver", "Heino",
    ];

    fn random_match_result() -> (u8, u8) {
        use std::collections::hash_map::RandomState;
        use std::hash::BuildHasher;

        let random = RandomState::new().hash_one(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        );

        match random % 10 {
            0 => (2, 0),
            1 => (2, 1),
            2 => (0, 2),
            3 => (1, 2),
            4 => (1, 1),
            5 => (0, 0),
            6 => (2, 0),
            7 => (2, 1),
            8 => (0, 2),
            9 => (1, 2),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_initial_pairing_and_second_round() {
        let pairing_engine = SwissPairing::<SimplePlayer>::new();

        // 8 players for a Magic draft pod
        let players: Vec<SimplePlayer> = TEST_PLAYERS
            .iter()
            .map(|p| SimplePlayer(p.to_string()))
            .collect();

        let mut round1_matches: Vec<SimpleMatch<SimplePlayer>> = Vec::with_capacity(4);

        (0..4).for_each(|i| {
            let (p1_wins, p2_wins) = random_match_result();
            round1_matches.push(SimpleMatch {
                player1: players[i * 2].clone(),
                player2: Some(players[i * 2 + 1].clone()),
                player1_wins: Some(p1_wins),
                player2_wins: Some(p2_wins),
                is_bye: false,
                round: 1,
            });
        });

        println!("Round 1 Results:");
        for game in &round1_matches {
            println!("{game}");
        }

        let mut round2_pairings: Vec<SimpleMatch<SimplePlayer>> =
            pairing_engine.pair(&round1_matches, 2);

        assert_eq!(round2_pairings.len(), 4);

        // Check that players are not paired against previous opponents
        for game in &round2_pairings {
            if let Some(p2) = &game.player2 {
                // Find if these players played before
                let played_before = round1_matches.iter().any(|m| {
                    (m.player1 == game.player1 && m.player2.as_ref() == Some(p2))
                        || (m.player1 == *p2 && m.player2.as_ref() == Some(&game.player1))
                });
                assert!(
                    !played_before,
                    "Players {:?} and {:?} already played",
                    game.player1, p2
                );
            }
        }

        round2_pairings[0].player1_wins = Some(1);
        round2_pairings[0].player2_wins = Some(2);

        round2_pairings[1].player1_wins = Some(2);
        round2_pairings[1].player2_wins = Some(0);

        round2_pairings[2].player1_wins = Some(2);
        round2_pairings[2].player2_wins = Some(1);

        round2_pairings[3].player1_wins = Some(1);
        round2_pairings[3].player2_wins = Some(1);

        println!("Round 2 Pairings:");
        for game in &round2_pairings {
            println!("{game}");
        }

        let mut all_matches = round1_matches.clone();
        all_matches.extend(round2_pairings);

        let round3_pairings: Vec<SimpleMatch<SimplePlayer>> = pairing_engine.pair(&all_matches, 3);

        println!("Round 3 Pairings:");
        for game in &round3_pairings {
            println!("{game}");
        }
        assert_eq!(round3_pairings.len(), 4);

        // Check that players are not paired against previous opponents
        for game in &round3_pairings {
            if let Some(p2) = &game.player2 {
                // Find if these players played before
                let played_before = round1_matches.iter().any(|m| {
                    (m.player1 == game.player1 && m.player2.as_ref() == Some(p2))
                        || (m.player1 == *p2 && m.player2.as_ref() == Some(&game.player1))
                });
                assert!(
                    !played_before,
                    "Players {:?} and {:?} already played",
                    game.player1, p2
                );
            }
        }
    }

    #[test]
    fn test_odd_number_of_players() {
        let pairing_engine = SwissPairing::<SimplePlayer>::new();

        // 7 players - one will get a bye
        let players: Vec<SimplePlayer> = TEST_PLAYERS[..7]
            .iter()
            .map(|p| SimplePlayer(p.to_string()))
            .collect();

        let mut matches = vec![];

        // Create round 1 results with random outcomes
        for i in 0..3 {
            let (p1_wins, p2_wins) = random_match_result();
            matches.push(SimpleMatch {
                player1: players[i * 2].clone(),
                player2: Some(players[i * 2 + 1].clone()),
                player1_wins: Some(p1_wins),
                player2_wins: Some(p2_wins),
                is_bye: false,
                round: 1,
            });
        }

        // Player 7 gets a bye
        matches.push(SimpleMatch {
            player1: players[6].clone(),
            player2: None,
            player1_wins: None,
            player2_wins: None,
            is_bye: true,
            round: 1,
        });

        println!("Round 1 Results (7 players):");
        for game in &matches {
            println!("  {}", game);
        }

        let round2_pairings: Vec<SimpleMatch<SimplePlayer>> = pairing_engine.pair(&matches, 2);

        println!("\nRound 2 Pairings:");
        for game in &round2_pairings {
            println!("  {}", game);
        }

        // Should have 4 pairings (3 matches + 1 bye)
        assert_eq!(round2_pairings.len(), 4);

        // Exactly one bye
        let bye_count = round2_pairings.iter().filter(|m| m.is_bye).count();
        assert_eq!(bye_count, 1);

        // No repeat pairings
        for game in &round2_pairings {
            if let Some(p2) = &game.player2 {
                let played_before = matches.iter().any(|m| {
                    (m.player1 == game.player1 && m.player2.as_ref() == Some(p2))
                        || (m.player1 == *p2 && m.player2.as_ref() == Some(&game.player1))
                });
                assert!(!played_before);
            }
        }
    }
}
