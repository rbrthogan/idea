from idea.evolution_tree import build_evolution_tree


def test_unchanged_survivor_is_continuation_not_new_node():
    history = [
        [
            {
                "id": "seed-a",
                "title": "Seed A",
                "content": "A",
                "parent_ids": [],
                "island_id": 0,
            },
            {
                "id": "seed-b",
                "title": "Seed B",
                "content": "B",
                "parent_ids": [],
                "island_id": 1,
            },
        ],
        [
            {
                "id": "seed-a",
                "title": "Seed A",
                "content": "A",
                "parent_ids": [],
                "island_id": 0,
            },
            {
                "id": "child-c",
                "title": "Child C",
                "content": "C",
                "parent_ids": ["seed-a", "seed-b"],
                "island_id": 1,
            },
        ],
        [
            {
                "id": "seed-a",
                "title": "Seed A",
                "content": "A",
                "parent_ids": [],
                "island_id": 0,
            },
            {
                "id": "child-c",
                "title": "Child C",
                "content": "C",
                "parent_ids": ["seed-a", "seed-b"],
                "island_id": 1,
            },
        ],
    ]

    tree = build_evolution_tree(history)

    assert [node["id"] for node in tree["nodes"]] == ["seed-a", "seed-b", "child-c"]
    seed_node = tree["nodes_by_id"]["seed-a"]
    child_node = tree["nodes_by_id"]["child-c"]
    assert seed_node["first_generation"] == 0
    assert seed_node["last_generation"] == 2
    assert seed_node["continuation_generations"] == [1, 2]
    assert child_node["first_generation"] == 1
    assert child_node["continuation_generations"] == [2]
    assert {
        "id": "seed-a:0->1",
        "node_id": "seed-a",
        "from_generation": 0,
        "to_generation": 1,
    } in tree["continuations"]
    assert {
        "id": "seed-a:1->2",
        "node_id": "seed-a",
        "from_generation": 1,
        "to_generation": 2,
    } in tree["continuations"]


def test_tree_edges_classify_breeding_refinement_and_migration():
    history = [
        [
            {"id": "a", "title": "A", "content": "A", "parent_ids": [], "island_id": 0},
            {"id": "b", "title": "B", "content": "B", "parent_ids": [], "island_id": 1},
        ],
        [
            {
                "id": "child",
                "title": "Child",
                "content": "Child",
                "parent_ids": ["a", "b"],
                "island_id": 0,
            },
            {
                "id": "refined",
                "title": "Refined",
                "content": "Refined",
                "parent_ids": ["a"],
                "survivor_refined": True,
                "survivor_refined_from_id": "a",
                "island_id": 0,
            },
            {
                "id": "migrant",
                "title": "Migrant",
                "content": "Migrant",
                "parent_ids": ["b"],
                "migration_event": True,
                "migrated_from_id": "b",
                "island_id": 0,
                "origin_island_id": 1,
            },
        ],
    ]

    tree = build_evolution_tree(history)

    edge_types = {
        (edge["from"], edge["to"]): edge["type"]
        for edge in tree["edges"]
    }
    assert edge_types[("a", "child")] == "breeding"
    assert edge_types[("b", "child")] == "breeding"
    assert edge_types[("a", "refined")] == "survivor_refinement"
    assert edge_types[("b", "migrant")] == "migration"


def test_tree_reuses_inactive_lanes_instead_of_growing_by_total_nodes():
    history = [
        [
            {"id": "stable", "title": "Stable", "content": "A", "parent_ids": [], "island_id": 0},
            {"id": "line-0", "title": "Line 0", "content": "B", "parent_ids": [], "island_id": 1},
        ],
        [
            {"id": "stable", "title": "Stable", "content": "A", "parent_ids": [], "island_id": 0},
            {"id": "line-1", "title": "Line 1", "content": "C", "parent_ids": ["line-0"], "island_id": 1},
        ],
        [
            {"id": "stable", "title": "Stable", "content": "A", "parent_ids": [], "island_id": 0},
            {"id": "line-2", "title": "Line 2", "content": "D", "parent_ids": ["line-1"], "island_id": 1},
        ],
        [
            {"id": "stable", "title": "Stable", "content": "A", "parent_ids": [], "island_id": 0},
            {"id": "line-3", "title": "Line 3", "content": "E", "parent_ids": ["line-2"], "island_id": 1},
        ],
    ]

    tree = build_evolution_tree(history)

    assert len(tree["nodes"]) == 5
    assert tree["lane_count"] == 2
    assert tree["nodes_by_id"]["stable"]["lane_index"] == 0
    assert tree["nodes_by_id"]["line-0"]["lane_index"] == 1
    assert tree["nodes_by_id"]["line-1"]["lane_index"] == 1
    assert tree["nodes_by_id"]["line-2"]["lane_index"] == 1
    assert tree["nodes_by_id"]["line-3"]["lane_index"] == 1
    assert tree["generations"][2]["slots"] == [
        {"node_id": "stable", "kind": "continuation", "lane_index": 0},
        {"node_id": "line-2", "kind": "node", "lane_index": 1},
    ]


def test_tree_exposes_global_fitness_range_for_macro_shading():
    history = [
        [
            {
                "id": "low",
                "title": "Low",
                "content": "A",
                "parent_ids": [],
                "selection_metrics": {"fitness": 0.2},
            },
            {
                "id": "missing",
                "title": "Missing",
                "content": "B",
                "parent_ids": [],
            },
        ],
        [
            {
                "id": "high",
                "title": "High",
                "content": "C",
                "parent_ids": ["low"],
                "survival_metrics": {"survival_score": 0.9},
            },
        ],
    ]

    tree = build_evolution_tree(history)

    assert tree["fitness_range"] == {"min": 0.2, "max": 0.9}
    assert tree["nodes_by_id"]["low"]["fitness_value"] == 0.2
    assert tree["nodes_by_id"]["high"]["fitness_value"] == 0.9
    assert tree["nodes_by_id"]["missing"]["fitness_value"] is None


def test_tree_exposes_ancestor_and_descendant_sets_for_selection():
    history = [
        [
            {"id": "a", "title": "A", "content": "A", "parent_ids": [], "selection_metrics": {"fitness": 0.2}},
            {"id": "b", "title": "B", "content": "B", "parent_ids": [], "selection_metrics": {"fitness": 0.6}},
        ],
        [
            {"id": "c", "title": "C", "content": "C", "parent_ids": ["a", "b"], "selection_metrics": {"fitness": 0.8}},
            {"id": "side", "title": "Side", "content": "S", "parent_ids": ["b"], "selection_metrics": {"fitness": 0.4}},
        ],
        [
            {"id": "d", "title": "D", "content": "D", "parent_ids": ["c"], "selection_metrics": {"fitness": 0.9}},
        ],
    ]

    tree = build_evolution_tree(history)

    assert set(tree["nodes_by_id"]["c"]["ancestor_ids"]) == {"a", "b"}
    assert set(tree["nodes_by_id"]["c"]["descendant_ids"]) == {"d"}
    assert set(tree["nodes_by_id"]["b"]["descendant_ids"]) == {"c", "side", "d"}
    assert tree["nodes_by_id"]["side"]["ancestor_ids"] == ["b"]


def test_tree_exposes_fitness_context_for_inspector():
    history = [
        [
            {
                "id": "a",
                "title": "A",
                "content": "A",
                "parent_ids": [],
                "selection_metrics": {"elo": 1400, "diversity": 0.1, "fitness": 0.2},
            },
            {
                "id": "b",
                "title": "B",
                "content": "B",
                "parent_ids": [],
                "selection_metrics": {"elo": 1600, "diversity": 0.2, "fitness": 0.6},
            },
        ],
        [
            {
                "id": "c",
                "title": "C",
                "content": "C",
                "parent_ids": ["a", "b"],
                "selection_metrics": {"elo": 1550, "diversity": 0.3, "fitness": 0.8},
            },
            {
                "id": "side",
                "title": "Side",
                "content": "S",
                "parent_ids": ["b"],
                "selection_metrics": {"elo": 1450, "diversity": 0.9, "fitness": 0.4},
            },
        ],
        [
            {
                "id": "d",
                "title": "D",
                "content": "D",
                "parent_ids": ["c"],
                "selection_metrics": {"elo": 1700, "diversity": 0.4, "fitness": 0.9},
            },
            {
                "id": "low",
                "title": "Low",
                "content": "L",
                "parent_ids": ["side"],
                "selection_metrics": {"elo": 1300, "diversity": 0.5, "fitness": 0.3},
            },
        ],
    ]

    tree = build_evolution_tree(history)

    context = tree["nodes_by_id"]["c"]["fitness_context"]
    assert context["generation"] == 1
    assert context["global_rank"] == 2
    assert context["global_population_size"] == 6
    assert context["history_rank"] == 1
    assert context["history_population_size"] == 4
    assert context["generation_rank"] == 1
    assert context["generation_population_size"] == 2
    assert round(context["history_mean"], 6) == 0.5
    assert round(context["delta_vs_history_mean"], 6) == 0.3
    assert context["parent_average"] == 0.4
    assert round(context["delta_vs_parent_average"], 6) == 0.4
    assert context["best_descendant_id"] == "d"
    assert round(context["delta_vs_best_descendant"], 6) == 0.1

    elo_context = tree["nodes_by_id"]["c"]["metric_contexts"]["elo"]
    assert elo_context["global_rank"] == 3
    assert elo_context["history_rank"] == 2
    assert elo_context["global_population_size"] == 6
    assert elo_context["history_population_size"] == 4

    diversity_context = tree["nodes_by_id"]["c"]["metric_contexts"]["diversity"]
    assert diversity_context["global_rank"] == 4
    assert diversity_context["history_rank"] == 2
    assert diversity_context["global_population_size"] == 6
    assert diversity_context["history_population_size"] == 4
