from __future__ import annotations

import asyncio
import random
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np

from idea.evolution_progress import ProgressEmitter
from idea.evolution_types import GenerationWorkState


class EvolutionOrchestrator:
    """Canonical evolution execution flow used by both new and resumed runs."""

    def __init__(
        self,
        engine: Any,
        progress_callback: Callable[[Dict[str, Any]], Awaitable[None]],
    ):
        self.engine = engine
        self.progress_callback = progress_callback
        self.progress = ProgressEmitter(engine)

    async def run(
        self,
        *,
        start_generation: int,
        mode: str,
        additional_generations: int = 0,
    ) -> None:
        try:
            self.engine.reset_stop_state()

            if additional_generations > 0:
                self.engine.generations += additional_generations
                print(
                    f"📈 Extended evolution by {additional_generations} generations. "
                    f"New total: {self.engine.generations}"
                )

            if mode == "new":
                await self._run_new(start_generation=start_generation)
                return

            if mode not in {"resume", "continue"}:
                raise ValueError(f"Unsupported orchestrator mode: {mode}")

            await self._run_resume(start_generation=start_generation)

        except Exception as exc:
            import traceback

            print(f"Error in orchestrator run: {exc}")
            print(traceback.format_exc())
            await self.engine.save_checkpoint(status="error")
            await self.progress.emit(
                self.progress_callback,
                {
                    "is_running": False,
                    "error": str(exc),
                    "is_resumable": True,
                    "checkpoint_id": self.engine.checkpoint_id,
                    "diversity_history": self.engine.diversity_history.copy()
                    if self.engine.diversity_history
                    else [],
                },
            )

    async def _run_new(self, *, start_generation: int) -> None:
        if not self.engine.checkpoint_id:
            from datetime import datetime

            self.engine.checkpoint_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        self.engine.current_generation = 0
        self.engine.population = []
        self.engine.specific_prompts = []
        self.engine.contexts = []
        self.engine.history = []
        self.engine.breeding_prompts = []
        self.engine.tournament_history = []
        self.engine.diversity_history = []

        print("Generating initial population (Generation 0)...")

        est_tournament_rounds = max(1, self.engine.tournament_rounds)
        steps_per_gen = self.engine.pop_size + est_tournament_rounds + 1
        total_steps = (2 * self.engine.pop_size) + (self.engine.generations * steps_per_gen)

        await self._seed_initial_population(total_steps=total_steps)
        if self.engine.stop_requested:
            return

        self.engine.history = [self.engine.population.copy()]
        await self.engine._calculate_and_store_diversity()

        token_counts = self.engine.get_total_token_count()
        current_cost = token_counts["cost"]["total_cost"]
        self.engine.avg_idea_cost = current_cost / max(1, self.engine.pop_size)

        total_ideas_to_generate = self.engine.pop_size * (self.engine.generations + 1)
        remaining_ideas = total_ideas_to_generate - self.engine.pop_size
        remaining_tournaments = self.engine.generations
        estimated_total_cost = current_cost + (
            remaining_ideas * self.engine.avg_idea_cost
        ) + (remaining_tournaments * self.engine.avg_tournament_cost)
        token_counts["cost"]["estimated_total_cost"] = estimated_total_cost

        base_payload = self.progress.base(
            current_generation=0, is_running=True, include_core_state=True
        )
        base_payload.update(
            {
                "progress": (2 * self.engine.pop_size / max(1, total_steps)) * 100,
                "token_counts": token_counts,
            }
        )
        await self.progress.emit(self.progress_callback, base_payload)

        if self.engine.check_budget():
            self.engine.stop_requested = True
            self.engine.is_stopped = True
            stop_payload = self.progress.base(
                current_generation=0,
                is_running=False,
                include_core_state=True,
            )
            stop_payload.update(
                {
                    "is_stopped": True,
                    "stop_message": (
                        "Evolution stopped: Budget limit reached "
                        f"(${current_cost:.2f} / ${self.engine.max_budget:.2f})"
                    ),
                    "token_counts": token_counts,
                }
            )
            await self.progress.emit(self.progress_callback, stop_payload)
            return

        await asyncio.sleep(0.1)

        work_state = GenerationWorkState(
            generation_index=0,
            start_generation=start_generation,
            total_generations=self.engine.generations,
            total_steps=total_steps,
            steps_per_generation=steps_per_gen,
        )

        await self._evolve_generations(
            start_generation=0,
            total_steps=total_steps,
            step_offset=(2 * self.engine.pop_size),
            work_state=work_state,
            resuming=False,
        )

    async def _run_resume(self, *, start_generation: int) -> None:
        if start_generation == 0 and len(self.engine.population) < self.engine.pop_size:
            print(
                f"⚠️ Generation 0 incomplete ({len(self.engine.population)}/{self.engine.pop_size} ideas). "
                "Restarting seeding..."
            )
            await self._complete_initial_seeding()
            if self.engine.stop_requested:
                return
            start_generation = self.engine.current_generation

        print(
            f"🔄 Resuming evolution from generation {start_generation}/{self.engine.generations}"
        )

        token_counts = self.engine.get_total_token_count()
        initial_payload = self.progress.base(
            current_generation=start_generation,
            is_running=True,
            include_core_state=True,
        )
        initial_payload.update(
            {
                "is_resuming": True,
                "progress": (start_generation / self.engine.generations) * 100
                if self.engine.generations > 0
                else 0,
                "status_message": f"Resuming from generation {start_generation}...",
                "token_counts": token_counts,
            }
        )
        await self.progress.emit(self.progress_callback, initial_payload)

        est_tournament_rounds = max(1, self.engine.tournament_rounds)
        steps_per_gen = self.engine.pop_size + est_tournament_rounds + 1
        remaining_gens = self.engine.generations - start_generation
        total_steps = max(1, remaining_gens * steps_per_gen)

        work_state = GenerationWorkState(
            generation_index=start_generation,
            start_generation=start_generation,
            total_generations=self.engine.generations,
            total_steps=total_steps,
            steps_per_generation=steps_per_gen,
        )

        await self._evolve_generations(
            start_generation=start_generation,
            total_steps=total_steps,
            step_offset=0,
            work_state=work_state,
            resuming=True,
        )

    async def _seed_initial_population(self, *, total_steps: int) -> None:
        async def generate_single_seed():
            context_pool = await asyncio.to_thread(
                self.engine.ideator.generate_context, self.engine.idea_type
            )
            idea_text, specific_prompt = await asyncio.to_thread(
                self.engine.ideator.generate_idea_from_context,
                context_pool,
                self.engine.idea_type,
            )
            return (
                context_pool,
                {"id": uuid.uuid4(), "idea": idea_text, "parent_ids": []},
                specific_prompt,
            )

        seed_tasks = [generate_single_seed for _ in range(self.engine.pop_size)]

        base_info = {
            "current_generation": 0,
            "total_generations": self.engine.generations,
            "is_running": True,
        }

        seed_results = await self.engine._run_batch_with_progress(
            tasks=seed_tasks,
            progress_callback=self.progress_callback,
            base_progress_info=base_info,
            start_step=0,
            total_steps=total_steps,
            description_template="Seeding idea {completed}/{total}...",
        )

        if self.engine.stop_requested:
            completed_results = [r for r in seed_results if r is not None]
            for context_pool, idea, prompt in completed_results:
                self.engine.contexts.append(context_pool)
                self.engine.population.append(idea)
                self.engine.specific_prompts.append(prompt)

            payload = self.progress.base(
                current_generation=0,
                is_running=False,
                include_core_state=False,
            )
            payload.update(
                {
                    "is_stopped": True,
                    "history": [self.engine.population] if self.engine.population else [],
                    "contexts": self.engine.contexts,
                    "specific_prompts": self.engine.specific_prompts,
                    "stop_message": "Evolution stopped during initial generation",
                }
            )
            await self.progress.emit(self.progress_callback, payload)
            return

        for context_pool, idea, prompt in seed_results:
            self.engine.contexts.append(context_pool)
            self.engine.population.append(idea)
            self.engine.specific_prompts.append(prompt)

        async def refine_single(idea):
            refined_idea = await asyncio.to_thread(
                self.engine.critic.refine, idea, self.engine.idea_type
            )
            formatted_idea = await asyncio.to_thread(
                self.engine.formatter.format_idea, refined_idea, self.engine.idea_type
            )
            return formatted_idea

        refine_tasks = [lambda i=idea: refine_single(i) for idea in self.engine.population]

        refined_results = await self.engine._run_batch_with_progress(
            tasks=refine_tasks,
            progress_callback=self.progress_callback,
            base_progress_info=base_info,
            start_step=self.engine.pop_size,
            total_steps=total_steps,
            description_template="Refining idea {completed}/{total}...",
        )

        if self.engine.stop_requested:
            for i, result in enumerate(refined_results):
                if result is not None:
                    self.engine.population[i] = result

            payload = self.progress.base(
                current_generation=0,
                is_running=False,
                include_core_state=False,
            )
            payload.update(
                {
                    "is_stopped": True,
                    "history": [self.engine.population],
                    "contexts": self.engine.contexts,
                    "specific_prompts": self.engine.specific_prompts,
                    "stop_message": "Evolution stopped during refinement",
                }
            )
            await self.progress.emit(self.progress_callback, payload)
            return

        self.engine.population = refined_results

    async def _complete_initial_seeding(self) -> None:
        existing_count = len(self.engine.population)
        needed_count = self.engine.pop_size - existing_count
        print(
            f"🌱 Completing initial seeding: {existing_count} existing, need {needed_count} more"
        )

        async def generate_single_seed():
            context_pool = await asyncio.to_thread(
                self.engine.ideator.generate_context, self.engine.idea_type
            )
            idea_text, specific_prompt = await asyncio.to_thread(
                self.engine.ideator.generate_idea_from_context,
                context_pool,
                self.engine.idea_type,
            )
            return (
                context_pool,
                {"id": uuid.uuid4(), "idea": idea_text, "parent_ids": []},
                specific_prompt,
            )

        for i in range(needed_count):
            if self.engine.stop_requested:
                print("Stop requested during seeding completion")
                return

            await self.progress.emit(
                self.progress_callback,
                {
                    "current_generation": 0,
                    "total_generations": self.engine.generations,
                    "is_running": True,
                    "status_message": (
                        f"Creating seed idea {existing_count + i + 1}/{self.engine.pop_size}..."
                    ),
                    "progress": ((existing_count + i) / self.engine.pop_size) * 50,
                },
            )

            context_pool, idea, prompt = await generate_single_seed()
            self.engine.contexts.append(context_pool)
            self.engine.population.append(idea)
            self.engine.specific_prompts.append(prompt)

        print("Refining initial population...")
        refined_population = []
        for i, idea in enumerate(self.engine.population):
            if self.engine.stop_requested:
                print("Stop requested during refinement")
                return

            await self.progress.emit(
                self.progress_callback,
                {
                    "current_generation": 0,
                    "total_generations": self.engine.generations,
                    "is_running": True,
                    "status_message": f"Refining idea {i + 1}/{len(self.engine.population)}...",
                    "progress": 50 + ((i + 1) / len(self.engine.population)) * 50,
                },
            )

            refined_idea = await asyncio.to_thread(
                self.engine.critic.refine, idea, self.engine.idea_type
            )
            formatted_idea = await asyncio.to_thread(
                self.engine.formatter.format_idea, refined_idea, self.engine.idea_type
            )
            refined_population.append(formatted_idea)

        self.engine.population = refined_population
        self.engine.history = [self.engine.population.copy()]

        await self.engine._calculate_and_store_diversity()

        self.engine.current_generation = 1
        await self.engine.save_checkpoint(status="in_progress")

        token_counts = self.engine.get_total_token_count()
        payload = self.progress.base(
            current_generation=0,
            is_running=True,
            include_core_state=True,
        )
        payload.update(
            {
                "progress": 100 / (self.engine.generations + 1),
                "status_message": "Generation 0 complete!",
                "token_counts": token_counts,
            }
        )
        await self.progress.emit(self.progress_callback, payload)

        print(f"✅ Initial seeding complete. Population size: {len(self.engine.population)}")

    async def _evolve_generations(
        self,
        *,
        start_generation: int,
        total_steps: int,
        step_offset: int,
        work_state: GenerationWorkState,
        resuming: bool,
    ) -> None:
        elite_idea = work_state.elite_idea
        elite_breeding_prompt = work_state.elite_breeding_prompt

        for gen in range(start_generation, self.engine.generations):
            if self.engine.stop_requested:
                self.engine.is_stopped = True
                self.engine.current_generation = gen
                print(f"Stop requested - evolution halted after generation {gen}")
                await self.engine.save_checkpoint(status="paused")

                payload = self.progress.base(
                    current_generation=gen,
                    is_running=False,
                    include_core_state=True,
                )
                payload.update(
                    {
                        "is_stopped": True,
                        "is_resumable": True,
                        "checkpoint_id": self.engine.checkpoint_id,
                        "progress": self._generation_progress(
                            gen=gen,
                            start_generation=start_generation,
                            total_steps=total_steps,
                            step_offset=step_offset,
                            step_in_generation=0,
                            steps_per_generation=work_state.steps_per_generation,
                        ),
                        "stop_message": (
                            f"Evolution paused after generation {gen}. You can resume this evolution."
                        ),
                        "token_counts": self.engine.get_total_token_count(),
                    }
                )
                await self.progress.emit(self.progress_callback, payload)
                return

            print(f"Starting generation {gen + 1}...")

            new_population: List[Any] = []
            generation_breeding_prompts: List[Optional[str]] = []
            random.shuffle(self.engine.population)

            elite_processed = False
            if elite_idea is not None:
                refined_elite = await asyncio.to_thread(
                    self.engine.critic.refine, elite_idea, self.engine.idea_type
                )
                formatted_elite = await asyncio.to_thread(
                    self.engine.formatter.format_idea,
                    refined_elite,
                    self.engine.idea_type,
                )

                if isinstance(formatted_elite, dict):
                    formatted_elite["elite_selected"] = True
                    formatted_elite["elite_source_id"] = elite_idea.get("id")
                    formatted_elite["elite_source_generation"] = gen
                else:
                    formatted_elite = {
                        "id": uuid.uuid4(),
                        "idea": formatted_elite,
                        "parent_ids": [],
                        "elite_selected": True,
                        "elite_source_id": elite_idea.get("id"),
                        "elite_source_generation": gen,
                    }

                new_population.append(formatted_elite)
                generation_breeding_prompts.append(elite_breeding_prompt)
                elite_processed = True

            current_pop_size = len(self.engine.population)
            ideas_to_breed = current_pop_size - (1 if elite_processed else 0)
            elite_idea = None
            elite_breeding_prompt = None

            tournament_start_cost = self.engine.get_total_token_count()["cost"]["total_cost"]

            current_gen_start_step = step_offset + (
                (gen - start_generation) * work_state.steps_per_generation
            )

            gen_base_progress_info = {
                "current_generation": gen + 1,
                "total_generations": self.engine.generations,
                "is_running": True,
            }

            loop = asyncio.get_running_loop()

            pairs_per_round = max(1, len(self.engine.population) // 2)
            total_pairs = pairs_per_round * max(1, self.engine.tournament_rounds)

            def thread_safe_callback(completed, _total):
                round_num = min(
                    self.engine.tournament_rounds,
                    (completed // pairs_per_round) + 1,
                )
                tournament_fraction = (completed / total_pairs) if total_pairs else 1.0

                async def send_update():
                    await self.progress_callback(
                        {
                            **gen_base_progress_info,
                            "progress": (
                                (
                                    current_gen_start_step
                                    + (
                                        tournament_fraction
                                        * max(1, self.engine.tournament_rounds)
                                    )
                                )
                                / max(1, total_steps)
                            )
                            * 100,
                            "status_message": (
                                f"Running Swiss round {round_num}/{self.engine.tournament_rounds}..."
                            ),
                        }
                    )

                asyncio.run_coroutine_threadsafe(send_update(), loop)

            tournament_rounds_details: List[Dict[str, Any]] = []
            global_ranks = await asyncio.to_thread(
                self.engine.critic.get_tournament_ranks,
                self.engine.population,
                self.engine.idea_type,
                self.engine.tournament_rounds,
                thread_safe_callback,
                tournament_rounds_details,
                self.engine.full_tournament_rounds,
            )
            self.engine._set_tournament_history(gen + 1, tournament_rounds_details)

            if self.engine.stop_requested:
                payload = self.progress.base(
                    current_generation=gen + 1,
                    is_running=False,
                    include_core_state=True,
                )
                payload.update(
                    {
                        "is_stopped": True,
                        "stop_message": "Evolution stopped during tournaments",
                    }
                )
                await self.progress.emit(self.progress_callback, payload)
                return

            tournament_end_cost = self.engine.get_total_token_count()["cost"]["total_cost"]
            current_tournament_cost = tournament_end_cost - tournament_start_cost
            if self.engine.avg_tournament_cost == 0:
                self.engine.avg_tournament_cost = current_tournament_cost
            else:
                self.engine.avg_tournament_cost = (
                    self.engine.avg_tournament_cost + current_tournament_cost
                ) / 2

            global_parent_slots = self.engine._allocate_parent_slots(global_ranks, ideas_to_breed)

            breeding_tasks_data = []
            for _ in range(ideas_to_breed):
                if global_parent_slots:
                    parent_indices = self.engine._select_parents_from_slots(
                        global_parent_slots, list(global_ranks.keys())
                    )
                    parent_ideas = [self.engine.population[idx] for idx in parent_indices]
                else:
                    parent_indices = np.random.choice(
                        list(global_ranks.keys()),
                        size=self.engine.breeder.parent_count,
                        replace=False,
                    )
                    parent_ideas = [self.engine.population[idx] for idx in parent_indices]
                breeding_tasks_data.append(parent_ideas)

            async def breed_single_child(parent_ideas):
                new_idea = await asyncio.to_thread(
                    self.engine.breeder.breed, parent_ideas, self.engine.idea_type
                )

                prompt = None
                if isinstance(new_idea, dict) and "specific_prompt" in new_idea:
                    prompt = new_idea["specific_prompt"]

                refined_idea = await asyncio.to_thread(
                    self.engine.critic.refine, new_idea, self.engine.idea_type
                )
                formatted_idea = await asyncio.to_thread(
                    self.engine.formatter.format_idea,
                    refined_idea,
                    self.engine.idea_type,
                )
                return formatted_idea, prompt

            breeding_tasks = [
                (lambda p=p: breed_single_child(p)) for p in breeding_tasks_data
            ]

            breeding_start_step = current_gen_start_step + max(1, self.engine.tournament_rounds)
            current_gen_base = 1 if elite_processed else 0

            breeding_results = await self.engine._run_batch_with_progress(
                tasks=breeding_tasks,
                progress_callback=self.progress_callback,
                base_progress_info=gen_base_progress_info,
                start_step=breeding_start_step + current_gen_base,
                total_steps=max(1, total_steps),
                description_template="Breeding and refining idea {completed}/{total}...",
            )

            if self.engine.stop_requested:
                completed_results = [r for r in breeding_results if r is not None]
                for idea, prompt in completed_results:
                    new_population.append(idea)
                    generation_breeding_prompts.append(prompt)

                if new_population:
                    self.engine.history.append(new_population)

                payload = self.progress.base(
                    current_generation=gen + 1,
                    is_running=False,
                    include_core_state=True,
                )
                payload.update(
                    {
                        "is_stopped": True,
                        "stop_message": "Evolution stopped during breeding",
                    }
                )
                await self.progress.emit(self.progress_callback, payload)
                return

            for result in breeding_results:
                if result is None:
                    generation_breeding_prompts.append(None)
                    continue

                idea, prompt = result
                new_population.append(idea)
                generation_breeding_prompts.append(prompt)

            token_counts = self.engine.get_total_token_count()
            current_cost = token_counts["cost"]["total_cost"]

            total_ideas = self.engine.pop_size * (self.engine.generations + 1)
            completed_ideas = self.engine.pop_size + (gen * self.engine.pop_size) + len(new_population)
            remaining_ideas_in_run = total_ideas - completed_ideas
            remaining_tournaments = self.engine.generations - 1 - gen

            estimated_total_cost = current_cost + (
                remaining_ideas_in_run * self.engine.avg_idea_cost
            ) + (remaining_tournaments * self.engine.avg_tournament_cost)
            token_counts["cost"]["estimated_total_cost"] = estimated_total_cost

            if self.engine.check_budget():
                self.engine.stop_requested = True
                self.engine.is_stopped = True
                if new_population:
                    self.engine.history.append(new_population)
                payload = self.progress.base(
                    current_generation=gen + 1,
                    is_running=False,
                    include_core_state=True,
                )
                payload.update(
                    {
                        "is_stopped": True,
                        "stop_message": (
                            "Evolution stopped: Budget limit reached "
                            f"(${current_cost:.2f} / ${self.engine.max_budget:.2f})"
                        ),
                        "token_counts": token_counts,
                    }
                )
                await self.progress.emit(self.progress_callback, payload)
                return

            history_copy = self.engine.history.copy()
            history_copy.append(new_population.copy())

            breeding_prompts_with_current = self.engine.breeding_prompts.copy()
            breeding_prompts_with_current.append(generation_breeding_prompts.copy())

            await self.progress.emit(
                self.progress_callback,
                {
                    "current_generation": gen + 1,
                    "total_generations": self.engine.generations,
                    "is_running": True,
                    "history": history_copy,
                    "contexts": self.engine.contexts,
                    "specific_prompts": self.engine.specific_prompts,
                    "breeding_prompts": breeding_prompts_with_current,
                    "progress": (
                        (
                            breeding_start_step
                            + current_gen_base
                            + len(new_population)
                        )
                        / max(1, total_steps)
                    )
                    * 100,
                    "token_counts": token_counts,
                },
            )

            await asyncio.sleep(0.1)

            self.engine.population = new_population
            self.engine.history.append(self.engine.population)
            self.engine.breeding_prompts.append(generation_breeding_prompts)

            await self.engine._calculate_and_store_diversity()

            await self._apply_oracle_if_enabled(gen)

            if gen < self.engine.generations - 1:
                elite_idea, elite_breeding_prompt = await self._select_elite_for_next_generation(gen)

            self.engine.current_generation = gen + 1

            checkpoint_status = (
                "in_progress" if gen < self.engine.generations - 1 else "complete"
            )
            checkpoint_path = await self.engine.save_checkpoint(status=checkpoint_status)
            if checkpoint_path:
                await self.progress.emit(
                    self.progress_callback,
                    {
                        "current_generation": gen + 1,
                        "total_generations": self.engine.generations,
                        "is_running": True,
                        "checkpoint_saved": True,
                        "checkpoint_id": self.engine.checkpoint_id,
                    },
                )

        if not self.engine.stop_requested:
            token_counts = self.engine.get_total_token_count()
            token_counts["cost"]["estimated_total_cost"] = token_counts["cost"]["total_cost"]
            payload = self.progress.base(
                current_generation=self.engine.generations,
                is_running=False,
                include_core_state=True,
            )
            payload.update(
                {
                    "progress": 100,
                    "token_counts": token_counts,
                }
            )
            await self.engine.save_checkpoint(status="complete")
            await self.progress.emit(self.progress_callback, payload)

    async def _apply_oracle_if_enabled(self, gen: int) -> None:
        if not self.engine.oracle:
            return

        try:
            oracle_result = self.engine.oracle.analyze_and_diversify(
                self.engine.history, self.engine.idea_type
            )
            replace_idx = await self.engine._find_least_interesting_idea_idx(self.engine.population)
            idea_prompt = oracle_result["idea_prompt"]
            prompts = self.engine._get_template_prompts()
            extended_prompt = idea_prompt
            if hasattr(prompts, "template") and prompts.template.special_requirements:
                extended_prompt = (
                    f"{idea_prompt}\n\nConstraints:\n{prompts.template.special_requirements}"
                )

            new_idea = self.engine.ideator.generate_text(extended_prompt)
            oracle_idea = {
                "id": uuid.uuid4(),
                "idea": new_idea,
                "parent_ids": [],
                "oracle_generated": True,
                "oracle_analysis": oracle_result["oracle_analysis"],
            }

            refined_oracle_idea = self.engine.critic.refine(oracle_idea, self.engine.idea_type)
            formatted_oracle_idea = self.engine.formatter.format_idea(
                refined_oracle_idea, self.engine.idea_type
            )

            if not formatted_oracle_idea.get("oracle_generated", False):
                formatted_oracle_idea["oracle_generated"] = True
                formatted_oracle_idea["oracle_analysis"] = oracle_idea.get(
                    "oracle_analysis", ""
                )

            old_idea = self.engine.population[replace_idx]
            old_idea_id = str(old_idea.get("id", "")) if isinstance(old_idea, dict) else ""
            if old_idea_id:
                await self.engine._remove_embedding(old_idea_id)

            self.engine.population[replace_idx] = formatted_oracle_idea
            if self.engine.breeding_prompts:
                self.engine.breeding_prompts[-1][replace_idx] = idea_prompt

            self.engine.history[-1] = self.engine.population.copy()

            token_counts = self.engine.get_total_token_count()
            await self.progress.emit(
                self.progress_callback,
                {
                    "current_generation": gen + 1,
                    "total_generations": self.engine.generations,
                    "is_running": True,
                    "history": self.engine.history,
                    "contexts": self.engine.contexts,
                    "specific_prompts": self.engine.specific_prompts,
                    "breeding_prompts": self.engine.breeding_prompts,
                    "progress": ((gen + 1) / max(1, self.engine.generations)) * 100,
                    "oracle_update": True,
                    "token_counts": token_counts,
                },
            )
        except Exception as exc:
            print(f"Oracle failed with error: {exc}. Continuing without Oracle enhancement.")

    async def _select_elite_for_next_generation(
        self, gen: int
    ) -> tuple[Optional[Any], Optional[str]]:
        elite_idea = None
        elite_breeding_prompt = None
        try:
            most_diverse_idx = await self.engine._find_most_diverse_idea_idx(self.engine.population)
            elite_idea = (
                self.engine.population[most_diverse_idx].copy()
                if isinstance(self.engine.population[most_diverse_idx], dict)
                else self.engine.population[most_diverse_idx]
            )

            if isinstance(self.engine.population[most_diverse_idx], dict):
                self.engine.population[most_diverse_idx]["elite_selected_source"] = True
                self.engine.population[most_diverse_idx]["elite_target_generation"] = gen + 1
                self.engine.history[-1] = self.engine.population.copy()

            if (
                self.engine.breeding_prompts
                and self.engine.breeding_prompts[-1]
                and most_diverse_idx < len(self.engine.breeding_prompts[-1])
            ):
                elite_breeding_prompt = self.engine.breeding_prompts[-1][most_diverse_idx]

            token_counts = self.engine.get_total_token_count()
            await self.progress.emit(
                self.progress_callback,
                {
                    "current_generation": gen + 1,
                    "total_generations": self.engine.generations,
                    "is_running": True,
                    "history": self.engine.history,
                    "contexts": self.engine.contexts,
                    "specific_prompts": self.engine.specific_prompts,
                    "breeding_prompts": self.engine.breeding_prompts,
                    "progress": ((gen + 1) / max(1, self.engine.generations)) * 100,
                    "elite_selection_update": True,
                    "token_counts": token_counts,
                },
            )
        except Exception as exc:
            print(f"Creative selection failed with error: {exc}. Continuing without creative selection.")

        return elite_idea, elite_breeding_prompt

    @staticmethod
    def _generation_progress(
        *,
        gen: int,
        start_generation: int,
        total_steps: int,
        step_offset: int,
        step_in_generation: int,
        steps_per_generation: int,
    ) -> float:
        current = step_offset + ((gen - start_generation) * steps_per_generation) + step_in_generation
        return (current / max(1, total_steps)) * 100
