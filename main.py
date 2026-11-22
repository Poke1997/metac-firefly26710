import argparse
import asyncio
import logging
from datetime import datetime, timezone
from typing import Literal

import os
from contextvars import ContextVar

import dotenv

import bot_config
import budget
import experiment
import run_log

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MonetaryCostManager,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# The slot the current forecast belongs to. Set per asyncio task, so forecasts
# running side by side each see their own slot.
_CURRENT_SLOT: ContextVar[experiment.Slot | None] = ContextVar("current_slot", default=None)


def _asknews_is_configured() -> bool:
    has_client = bool(os.getenv("ASKNEWS_CLIENT_ID")) and bool(os.getenv("ASKNEWS_SECRET"))
    return has_client or bool(os.getenv("ASKNEWS_API_KEY"))


def _forecast_llm(model: str, effort: str = "high") -> GeneralLlm:
    """
    A model set up to forecast. Temperature is left unset because the current
    Claude models do not accept it. The cache setting marks the first message
    (the shared question and research) as cacheable, so later forecasts on the
    same question read it at the cached price.
    """
    return GeneralLlm(
        model=model,
        reasoning_effort=effort,
        max_tokens=32000,
        timeout=900,
        allowed_tries=3,
        cache_control_injection_points=[{"location": "message", "index": 0}],
    )


def build_llms(tier: budget.Tier) -> dict[str, GeneralLlm]:
    """
    Models the forecasting-tools framework asks for by name. "default" comes
    from the budget tier; research, summarising and output parsing always use
    the helper model.
    """
    assert tier.forecaster_model is not None
    return {
        "default": _forecast_llm(tier.forecaster_model),
        "researcher": GeneralLlm(
            model=bot_config.HELPER_MODEL,
            web_search_options={"search_context_size": "medium"},
            max_tokens=16000,
            timeout=900,
            allowed_tries=3,
        ),
        "summarizer": GeneralLlm(
            model=bot_config.HELPER_MODEL, max_tokens=8000, timeout=300, allowed_tries=3
        ),
        "parser": GeneralLlm(
            model=bot_config.HELPER_MODEL, max_tokens=8000, timeout=300, allowed_tries=3
        ),
    }


def build_slot_llms(tier: budget.Tier) -> dict[str, GeneralLlm]:
    """Models the forecast slots use, keyed by experiment.Slot.llm_key."""
    assert tier.forecaster_model is not None
    return {
        "forecaster": _forecast_llm(tier.forecaster_model),
        "economy": _forecast_llm(bot_config.ECONOMY_MODEL),
        "helper": _forecast_llm(bot_config.HELPER_MODEL),
        "forecaster_medium": _forecast_llm(tier.forecaster_model, effort="medium"),
    }


def _write_github_output(name: str, value: str) -> None:
    output_path = os.getenv("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a") as output_file:
            output_file.write(f"{name}={value}\n")


class FireflyBot(ForecastBot):
    """
    This is the template bot for the Fall 2026 FutureEval Bot Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    This bot is identical to SummerTemplateBot2026.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name")
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    # Set in __main__ before the bot runs.
    slots: list[experiment.Slot] = []
    slot_llms: dict[str, GeneralLlm] = {}
    run_mode: str = "tournament"

    ##################################### SLOTS #####################################

    async def _research_and_make_predictions(self, question: MetaculusQuestion):
        """
        Same job as the framework version, but each forecast runs in a named
        slot (see experiment.py) and every slot's result is logged.

        Slots run in two waves. The first slot for each model goes first; the
        rest follow once it has finished, so they find the shared question and
        research already in that model's prompt cache.
        """
        from forecasting_tools.data_models.forecast_report import ResearchWithPredictions

        notepad = await self._get_notepad(question)
        notepad.total_research_reports_attempted += 1
        research = await self.run_research(question)
        summary_report = await self.summarize_research(question, research)
        research_to_use = (
            summary_report if self.use_research_summary_to_forecast else research
        )

        slots = self.slots or [experiment.Slot("control_1", True, "forecaster", "template")]
        first_wave: list[experiment.Slot] = []
        second_wave: list[experiment.Slot] = []
        models_started: set[str] = set()
        for slot in slots:
            if slot.llm_key in models_started:
                second_wave.append(slot)
            else:
                models_started.add(slot.llm_key)
                first_wave.append(slot)

        outcomes: dict[str, ReasonedPrediction | BaseException] = {}
        for wave in (first_wave, second_wave):
            if not wave:
                continue
            results = await asyncio.gather(
                *(self._predict_in_slot(question, research_to_use, slot) for slot in wave),
                return_exceptions=True,
            )
            for slot, result in zip(wave, results):
                if isinstance(result, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                    raise result
                outcomes[slot.name] = result

        slot_results = []
        predictions = []
        errors: list[str] = []
        for slot in slots:
            result = outcomes[slot.name]
            entry = {
                "slot": slot.name,
                "submitted": slot.submitted,
                "model": self.slot_llms[slot.llm_key].model if slot.llm_key in self.slot_llms else None,
                "llm_key": slot.llm_key,
                "variant": slot.variant,
            }
            if isinstance(result, BaseException):
                entry["error"] = f"{type(result).__name__}: {result}"
                if slot.submitted:
                    errors.append(f"{slot.name}: {entry['error']}")
                else:
                    logger.warning(f"Shadow forecast {slot.name} failed for {question.page_url}: {result}")
            else:
                entry["prediction"] = result.prediction_value
                entry["reasoning"] = result.reasoning
                if slot.submitted:
                    predictions.append(result)
            slot_results.append(entry)
        run_log.record_question(self.run_mode, question, research, slot_results)

        if errors:
            logger.warning(f"Encountered errors while predicting: {errors}")
        if not predictions:
            raise RuntimeError(
                f"Every submitted forecast failed for {question.page_url}: {errors}"
            )
        return ResearchWithPredictions(
            research_report=research,
            summary_report=summary_report,
            errors=errors,
            predictions=predictions,
        )

    async def _predict_in_slot(
        self, question: MetaculusQuestion, research: str, slot: experiment.Slot
    ) -> ReasonedPrediction:
        _CURRENT_SLOT.set(slot)
        return await self._make_prediction(question, research)

    async def _invoke_forecaster(self, prompt: str) -> str:
        """
        Send a forecast prompt to the current slot's model. The prompt goes out
        as two messages, shared context then instructions, so the context can be
        cached across the slots of one question.
        """
        slot = _CURRENT_SLOT.get()
        if slot is None or slot.llm_key not in self.slot_llms:
            return await self.get_llm("default", "llm").invoke(prompt)
        llm = self.slot_llms[slot.llm_key]
        context, instructions = experiment.split_prompt(prompt)
        instructions = experiment.apply_variant(instructions, slot.variant)
        if not context:
            return await llm.invoke(instructions)
        try:
            return await llm.invoke(
                [
                    {"role": "user", "content": context},
                    {"role": "user", "content": instructions},
                ]
            )
        except Exception as error:
            logger.warning(
                f"Two-message forecast call failed in slot {slot.name} ({error}); retrying as one message"
            )
            return await llm.invoke(context + instructions)

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Research = a web-search report from the "researcher" model, plus AskNews
        summaries when AskNews credentials are set. A failure in one source never
        blocks the other; if both fail the question is forecast without research.
        """
        async with self._concurrency_limiter:
            sections: list[str] = []

            researcher = self.get_llm("researcher")
            if isinstance(researcher, GeneralLlm):
                try:
                    report = await researcher.invoke(self._get_research_prompt(question))
                    sections.append(f"## Web research\n{report}")
                except Exception as error:
                    logger.error(f"Web research failed for {question.page_url}: {error}")

            if _asknews_is_configured():
                try:
                    news = await AskNewsSearcher().call_preconfigured_version(
                        "asknews/news-summaries", question.question_text
                    )
                    sections.append(f"## News summaries (AskNews)\n{news}")
                except Exception as error:
                    logger.error(f"AskNews failed for {question.page_url}: {error}")

            research = "\n\n".join(sections)
            if not research:
                logger.warning(f"No research found for {question.page_url}")
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    @staticmethod
    def _get_research_prompt(question: MetaculusQuestion) -> str:
        prompt = clean_indents(
            f"""
            You are a research assistant to a superforecaster. Today is {datetime.now(timezone.utc).strftime("%Y-%m-%d")}.
            The superforecaster will forecast the question below. Search the web and write a concise, factual briefing. You do not produce a forecast yourself.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}

            Cover, in this order:
            1. Current status: the latest facts that bear on the resolution criteria, each with its date. Say plainly whether the criteria look met, not met, or undecided on current information.
            2. The source the question will resolve from, if one is named: what it shows right now (the latest published value or statement).
            3. Base rate: how often comparable events have happened over comparable periods, or how the relevant number has moved historically, with figures.
            4. Scheduled events before the resolution date that could change the outcome.
            5. What prediction markets or other forecasters say, if you find any, with the numbers.
            6. The strongest evidence on each side.

            Prefer primary and recent sources. Give dates for facts. If you could not find something, say so instead of guessing.
            """
        )
        return prompt

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self._invoke_forecaster(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a probability forecast for a binary question.
            {self._create_resolved_question_parsing_message()}
            """
        )
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            {self._create_resolved_question_parsing_message()}
            """
        )
        reasoning = await self._invoke_forecaster(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._invoke_forecaster(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._invoke_forecaster(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_resolved_question_parsing_message(self) -> str:
        return "- If the text concludes that the question has already resolved *in the past* (i.e. it treats the question as decided rather than something to forecast), please DO NOT return a parsed output, even if a final forecast is also given. Instead indicate that the answer is not explicitly given in the text.\n"

    def _create_single_distribution_parsing_message(
        self, question: NumericQuestion | DateQuestion
    ) -> str:
        message = (
            "- The text may contain multiple percentile distributions (e.g. forecasts for several related questions/entities, or intermediate drafts before a final answer). You must return exactly ONE distribution: the single final distribution that answers the question stated above.\n"
            "- Never merge or concatenate percentile lists that refer to different entities, options, or scenarios. Each percentile should appear at most once in your output.\n"
            "- If there are multiple final distributions and you cannot tell which one answers the question stated above, do not guess or combine them. Instead indicate that the answer is not explicitly given in the text."
        )
        if question.group_question_option is not None:
            message += (
                f'\n- This question is specifically about "{question.group_question_option}" (one subquestion within a group of related questions). '
                f'If the text gives distributions for multiple subjects, return only the distribution for "{question.group_question_option}".'
            )
        return message

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    # Budget guard: pick the model tier from what the bot has already spent in
    # this credit cycle, and stop before the cycle budget is exceeded.
    spent_before = budget.load_spent()
    tier = budget.pick_tier(spent_before)
    run_cap = budget.run_cap_usd(spent_before)
    logger.info(
        f"Budget: ${spent_before:.2f} of ${budget.budget_usd():.2f} spent since "
        f"{budget.cycle_start().isoformat()}; tier '{tier.name}'; this run may spend up to ${run_cap:.2f}"
    )
    if tier.forecaster_model is None:
        logger.error(
            "Cycle budget is used up, so this run forecasts nothing. "
            "Raise BUDGET_USD or wait for the next credit cycle."
        )
        _write_github_output("ledger_changed", "false")
        run_log.record_status(
            run_mode,
            {
                "tier": tier.name,
                "questions_forecast": 0,
                "errors": ["cycle budget used up; nothing forecast"],
                "cycle_spent_usd": round(spent_before, 4),
                "cycle_budget_usd": budget.budget_usd(),
                "cycle_start": budget.cycle_start().isoformat(),
            },
        )
        raise SystemExit(0)

    bot = FireflyBot(
        research_reports_per_question=1,
        predictions_per_research_report=tier.predictions_per_report,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms=build_llms(tier),
    )
    bot.slots = experiment.build_slots(tier)
    bot.slot_llms = build_slot_llms(tier)
    bot.run_mode = run_mode
    logger.info("Slots: " + ", ".join(
        f"{slot.name}[{slot.llm_key}/{slot.variant}{'' if slot.submitted else '/shadow'}]" for slot in bot.slots
    ))

    # Per-mode tournament URL shown in the summary banner footer. These
    # piggyback on the forecasting_tools SDK constants and need updating
    # whenever those rotate seasons.
    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below. Everything runs inside the cost manager, which
    # raises once the run cap is reached.
    client = MetaculusClient()
    forecast_reports: list = []
    run_error: BaseException | None = None
    with MonetaryCostManager(run_cap) as cost_manager:
        try:
            if run_mode == "tournament":
                forecast_reports += asyncio.run(
                    bot.forecast_on_tournament(
                        client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
                    )
                )
                if os.getenv("INCLUDE_MINIBENCH", "true").lower() != "false":
                    forecast_reports += asyncio.run(
                        bot.forecast_on_tournament(
                            client.CURRENT_MINIBENCH_ID, return_exceptions=True
                        )
                    )
            elif run_mode == "metaculus_cup":
                # The Metaculus Cup may be uninitialized near the start of a season
                # (Jan/May/Sep). MetaculusClient.ACX_2025_TOURNAMENT = 32564 and
                # MetaculusClient.AI_2027_TOURNAMENT_ID = "ai-2027" are also valid
                # targets here.
                bot.skip_previously_forecasted_questions = False
                forecast_reports += asyncio.run(
                    bot.forecast_on_tournament(
                        client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
                    )
                )
            elif run_mode == "test_questions":
                # The bot-testing-area tournament contains all question types and is
                # the recommended target for smoke-testing your bot.
                # https://www.metaculus.com/tournament/bot-testing-area/
                bot.skip_previously_forecasted_questions = False
                forecast_reports += asyncio.run(
                    bot.forecast_on_tournament(
                        "bot-testing-area", return_exceptions=True
                    )
                )
        except BaseException as error:  # the ledger must be written even if the run dies
            run_error = error
        run_cost = cost_manager.current_usage

    spent_after = budget.record_spend(run_cost) if run_cost > 0 else spent_before
    _write_github_output("ledger_changed", "true" if run_cost > 0 else "false")
    logger.info(
        f"Budget: this run cost ${run_cost:.2f}; ${spent_after:.2f} of "
        f"${budget.budget_usd():.2f} spent this cycle"
    )
    report_errors = [
        f"{type(report).__name__}: {report}"[:500]
        for report in forecast_reports
        if isinstance(report, BaseException)
    ]
    if run_error is not None:
        report_errors.append(f"run stopped: {type(run_error).__name__}: {run_error}"[:500])
    run_log.record_status(
        run_mode,
        {
            "tier": tier.name,
            "forecaster_model": tier.forecaster_model,
            "slots": [slot.name for slot in bot.slots],
            "questions_forecast": sum(
                1 for report in forecast_reports if not isinstance(report, BaseException)
            ),
            "errors": report_errors,
            "run_cost_usd": round(run_cost, 4),
            "cycle_spent_usd": round(spent_after, 4),
            "cycle_budget_usd": budget.budget_usd(),
            "cycle_start": budget.cycle_start().isoformat(),
        },
    )
    if run_error is not None:
        raise run_error

    bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
