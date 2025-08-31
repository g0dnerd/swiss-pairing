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
    opponents: Vec<usize>,
    games_played: usize,
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
        let matching = self.maximum_weight_matching(&graph, players.len());

        self.create_pairings_from_matching(&matching, &players, round)
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

        players.into_iter().collect()
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
                stats[p1_idx].opponents.push(p2_idx);
                stats[p2_idx].opponents.push(p1_idx);

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

    fn build_pairing_graph(&self, players: &[P], stats: &[PlayerStats]) -> Vec<Vec<i32>> {
        let n = players.len();
        let mut graph = vec![vec![0i32; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                if stats[i].opponents.contains(&j) {
                    graph[i][j] = i32::MIN;
                } else {
                    // TODO: Tiebreakers
                    let point_diff = stats[i].points.abs_diff(stats[j].points);
                    graph[i][j] = i32::from(1000 - point_diff * 100);
                }
                graph[j][i] = graph[i][j];
            }
        }

        graph
    }

    fn maximum_weight_matching(&self, graph: &[Vec<i32>], n: usize) -> Vec<Option<usize>> {
        let mut matching = vec![None; n];
        let mut used = vec![false; n];

        let mut scores = vec![0i32; n];
        for i in 0..n {
            scores[i] = graph[i].iter().filter(|&&w| w > 0).sum();
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by_key(|&i| -scores[i]);

        for &i in &indices {
            if used[i] {
                continue;
            }

            let mut best_j = None;
            let mut best_weight = i32::MIN;

            (0..n).for_each(|j| {
                if i != j && !used[j] && graph[i][j] > best_weight {
                    best_weight = graph[i][j];
                    best_j = Some(j);
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
            let is_bye = player2.is_some();

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

    #[test]
    fn test_initial_pairing_and_second_round() {
        let pairing_engine = SwissPairing::<SimplePlayer>::new();

        // 8 players for a Magic draft pod
        let players: Vec<SimplePlayer> =
            (TEST_PLAYERS).map(|p| SimplePlayer(p.to_string())).to_vec();

        let game1: SimpleMatch<SimplePlayer> = SimpleMatch {
            player1: players[0].clone(),
            player2: Some(players[1].clone()),
            player1_wins: Some(2),
            player2_wins: Some(0),
            is_bye: false,
            round: 1,
        };

        let game2: SimpleMatch<SimplePlayer> = SimpleMatch {
            player1: players[2].clone(),
            player2: Some(players[3].clone()),
            player1_wins: Some(2),
            player2_wins: Some(1),
            is_bye: false,
            round: 1,
        };

        let game3: SimpleMatch<SimplePlayer> = SimpleMatch {
            player1: players[4].clone(),
            player2: Some(players[5].clone()),
            player1_wins: Some(1),
            player2_wins: Some(2),
            is_bye: false,
            round: 1,
        };

        let game4: SimpleMatch<SimplePlayer> = SimpleMatch {
            player1: players[6].clone(),
            player2: Some(players[7].clone()),
            player1_wins: Some(2),
            player2_wins: Some(0),
            is_bye: false,
            round: 1,
        };

        let round1_matches = vec![game1, game2, game3, game4];

        println!("Round 1 Pairings:");
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

        let round3_pairings: Vec<SimpleMatch<SimplePlayer>> =
            pairing_engine.pair(&round2_pairings, 3);

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
}
