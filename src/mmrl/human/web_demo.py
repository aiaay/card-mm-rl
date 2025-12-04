"""Interactive Streamlit demo for manually playing the card market env."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from mmrl.env.cards import get_rank_str
from mmrl.env.single_env import SingleCardEnv

CARD_SUITS = ["♠", "♥", "♦", "♣"]
SUIT_COLORS = {"♠": "black", "♣": "black", "♥": "red", "♦": "red"}

CARD_CSS = """
<style>
.card-grid {
    display: flex;
    gap: 0.8rem;
    margin-bottom: 1.0rem;
}
.card {
    width: 90px;
    height: 130px;
    border-radius: 12px;
    padding: 8px;
    border: 3px solid #ffffff22;
    background: linear-gradient(160deg, #1b1f3a, #222b45);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 6px 25px rgba(0,0,0,0.35);
}
.card.red {
    border-color: #f94144;
    color: #ffd9da;
}
.card.black {
    border-color: #577590;
    color: #f8f9fa;
}
.card .rank {
    font-size: 2.3rem;
    font-weight: 700;
    text-align: left;
}
.card .suit {
    font-size: 2.0rem;
    text-align: right;
}
.tag {
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
    margin-right: 0.4rem;
    background: #2b3955;
    color: #d7e3ff;
    border: 1px solid #3e4c69;
}
</style>
"""

DEFAULT_CFG = {
    "W0": 500.0,
    "episode_length": 10,
    "alpha": 0.3,
    "event_persist": 0.2,
    "events": {
        "none": 0.5,
        "ge10_only": 0.2,
        "le7_only": 0.1,
        "even_only": 0.1,
        "remap_value": 0.1,
    },
    "flags": {
        "enable_events": True,
        "enable_impact": True,
    },
    "spread": {"beta": 0.25},
}


def format_money(value: float) -> str:
    """Convert a float into a printable currency string."""
    return f"${value:,.2f}"


def build_cfg(
    base: Dict[str, Any],
    *,
    events: bool,
    impact: bool,
    alpha: float,
    episode_length: int,
    initial_wealth: float,
) -> Dict[str, Any]:
    """Create an env config with the user supplied overrides."""
    cfg = {
        **base,
        "W0": initial_wealth,
        "alpha": alpha,
        "episode_length": episode_length,
        "flags": {
            "enable_events": events,
            "enable_impact": impact,
        },
    }
    return cfg


def ensure_session_state():
    """Seed default keys in st.session_state."""
    defaults = {
        "env": None,
        "obs": None,
        "info": None,
        "cfg": DEFAULT_CFG,
        "done": True,
        "history": [],
        "last_result": None,
        "round_snapshot": None,
        "seed": 42,
        "last_cards": [],
        "last_depths": None,
        "last_impact": 0.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def snapshot_env(env: SingleCardEnv) -> Dict[str, Any]:
    """Capture the data we want to display for the current round."""
    event_dict = env.current_event.to_dict() if env.current_event else {"type": "none"}
    quote = env.quote
    return {
        "round": env.t,
        "total_rounds": env.T,
        "hints": list(env.hints),
        "event": event_dict,
        "wallet": env.W,
        "quote_bid": getattr(quote, "bid", 0.0),
        "quote_ask": getattr(quote, "ask", 0.0),
        "quote_mid": getattr(quote, "mid", 0.0),
        "spread": getattr(quote, "spread", 0.0),
        "depth_bid": env.disp_depths[0] if env.disp_depths else 0.0,
        "depth_ask": env.disp_depths[1] if env.disp_depths else 0.0,
        "mu": env.mu,
        "sigma": env.sigma,
        "hidden_cards": list(env.hidden_cards),
        "true_depths": tuple(env.true_depths),
    }


def describe_action(action_id: int) -> str:
    """Map discrete action id to a human-readable label."""
    if action_id == 0:
        return "Pass"
    if 1 <= action_id <= 10:
        return f"Buy {action_id}"
    if 11 <= action_id <= 20:
        return f"Sell {action_id - 10}"
    return f"Action {action_id}"


def render_hint_cards(hints: List[int]):
    """Display the active hints as styled playing cards."""
    if not hints:
        st.info("No hints this round. Time to read the tape.")
        return

    cards_html = '<div class="card-grid">'
    for idx, rank in enumerate(hints):
        suit = CARD_SUITS[idx % len(CARD_SUITS)]
        rank_text = get_rank_str(rank)
        color = SUIT_COLORS[suit]
        card_class = "red" if color == "red" else "black"
        cards_html += (
            f'<div class="card {card_class}">'
            f'<div class="rank">{rank_text}</div>'
            f'<div class="suit">{suit}</div>'
            "</div>"
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def init_game(cfg: Dict[str, Any], seed: int):
    """Instantiate a fresh environment and reset session state."""
    env = SingleCardEnv(cfg)
    obs, info = env.reset(seed=seed)
    st.session_state.env = env
    st.session_state.obs = obs
    st.session_state.info = info
    st.session_state.done = False
    st.session_state.history = []
    st.session_state.last_result = None
    st.session_state.round_snapshot = snapshot_env(env)
    st.session_state.seed = seed
    st.session_state.cfg = cfg


def perform_action(action_id: int):
    """Execute the selected action and store the result."""
    env: Optional[SingleCardEnv] = st.session_state.env
    if env is None or st.session_state.done:
        return

    cards_before = list(env.hidden_cards)
    depths_before = tuple(env.true_depths)
    pre_snapshot = snapshot_env(env)
    obs, reward, terminated, truncated, info = env.step(action_id)
    exec_price = info.get("exec_price", np.nan)
    price_impact = 0.0
    if 1 <= action_id <= 10:
        price_impact = exec_price - pre_snapshot["quote_ask"]
    elif 11 <= action_id <= 20:
        price_impact = pre_snapshot["quote_bid"] - exec_price

    entry = {
        "round": pre_snapshot["round"] + 1,
        "action": describe_action(action_id),
        "reward": reward,
        "exec_price": exec_price,
        "true_sum": info.get("true_sum", np.nan),
        "wealth_after": env.W,
        "event": pre_snapshot["event"].get("type", "none"),
        "cards": cards_before,
        "true_depths": depths_before,
        "price_impact": price_impact,
    }
    st.session_state.history.append(entry)
    st.session_state.last_result = entry
    st.session_state.last_cards = cards_before
    st.session_state.last_depths = depths_before
    st.session_state.last_impact = price_impact

    st.session_state.obs = obs
    st.session_state.info = info
    st.session_state.done = terminated or truncated

    if st.session_state.done:
        st.session_state.round_snapshot = pre_snapshot
    else:
        st.session_state.round_snapshot = snapshot_env(env)


def sidebar_controls() -> Dict[str, Any]:
    """Render all configuration widgets in the sidebar."""
    st.sidebar.header("Settings")
    events = st.sidebar.toggle("Enable Events", value=True)
    impact = st.sidebar.toggle("Enable Impact", value=True)
    episode_length = st.sidebar.slider("Episode Length", min_value=5, max_value=20, value=10)
    alpha = st.sidebar.slider(
        "Impact Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )
    initial_wealth = st.sidebar.number_input(
        "Initial Wealth (W0)",
        min_value=100.0,
        max_value=2000.0,
        value=500.0,
        step=50.0,
    )
    seed = st.sidebar.number_input(
        "Seed",
        min_value=0,
        max_value=10_000_000,
        value=int(st.session_state.get("seed", 42)),
        step=1,
    )

    cfg = build_cfg(
        DEFAULT_CFG,
        events=events,
        impact=impact,
        alpha=alpha,
        episode_length=episode_length,
        initial_wealth=initial_wealth,
    )

    if st.sidebar.button("Start / Reset Game", use_container_width=True):
        init_game(cfg, int(seed))

    return cfg


def render_market_view(snapshot: Dict[str, Any]):
    """Show the per-round market state (hints, quotes, event, etc.)."""
    round_id = snapshot["round"] + 1
    st.subheader(f"Round {round_id}/{snapshot['total_rounds']}")
    st.caption("Choose an action below to trade against the quote.")

    cards_col, info_col = st.columns([2, 3])
    with cards_col:
        st.markdown(CARD_CSS, unsafe_allow_html=True)
        render_hint_cards(snapshot["hints"])

    with info_col:
        bid_ask = f"{snapshot['quote_bid']:.2f} / {snapshot['quote_ask']:.2f}"
        depths = f"{snapshot['depth_bid']:.1f} / {snapshot['depth_ask']:.1f}"
        st.metric("Wallet", format_money(snapshot["wallet"]))
        st.metric("Quote (Bid / Ask)", bid_ask)
        st.metric("Displayed Depth (Bid / Ask)", depths)

        mu_sigma = f"μ = {snapshot['mu']:.2f}, σ = {snapshot['sigma']:.2f}"
        st.write(f"Posterior: {mu_sigma}")

    event_type = snapshot["event"].get("type", "none")
    if event_type != "none":
        extra = ", ".join(
            f"{k}={v}" for k, v in snapshot["event"].items() if k != "type"
        )
        event_label = f"Event: **{event_type}**  {extra}"
    else:
        event_label = "Event: None"
    st.markdown(f"<span class='tag'>{event_label}</span>", unsafe_allow_html=True)


def render_action_panel(mask: np.ndarray):
    """Render large buttons for pass/buy/sell actions."""
    st.markdown("### Actions")
    if st.session_state.done:
        st.warning("Round complete. Reset from the sidebar to play again.")
        return

    cols = st.columns([1, 2, 2])
    disabled = st.session_state.done

    with cols[0]:
        if st.button(
            "Pass",
            type="secondary",
            use_container_width=True,
            disabled=disabled or not mask[0],
        ):
            perform_action(0)

    with cols[1]:
        st.caption("Buy (1-10)")
        buy_cols = st.columns(5)
        for size in range(1, 11):
            col = buy_cols[(size - 1) % 5]
            with col:
                action_id = size
                if st.button(
                    f"{size}",
                    use_container_width=True,
                    disabled=disabled or not mask[action_id],
                    key=f"buy_{size}",
                ):
                    perform_action(action_id)

    with cols[2]:
        st.caption("Sell (1-10)")
        sell_cols = st.columns(5)
        for size in range(1, 11):
            col = sell_cols[(size - 1) % 5]
            action_id = size + 10
            with col:
                if st.button(
                    f"{size}",
                    use_container_width=True,
                    disabled=disabled or not mask[action_id],
                    key=f"sell_{size}",
                ):
                    perform_action(action_id)


def render_history():
    """Display the dataframe of completed rounds."""
    st.markdown("### Round History")
    if not st.session_state.history:
        st.info("Play a round to see the trade log.")
        return

    history_df = pd.DataFrame(st.session_state.history)
    history_df["reward"] = history_df["reward"].map(lambda x: round(x, 2))
    history_df["exec_price"] = history_df["exec_price"].map(lambda x: round(x, 2))
    history_df["wealth_after"] = history_df["wealth_after"].map(lambda x: round(x, 2))
    st.dataframe(history_df, use_container_width=True)


def main():
    """Entrypoint for the Streamlit script."""
    st.set_page_config(page_title="MMRL Card Market Demo", layout="wide")
    ensure_session_state()

    st.title("Card Market RL — Interactive Demo")
    st.write(
        "Trade against the market maker, see your hints rendered as cards, "
        "and step through each round of the episode. "
        "Use the sidebar to toggle events, impact, and other environment settings."
    )

    sidebar_controls()

    env: Optional[SingleCardEnv] = st.session_state.env

    if env is None:
        st.info("Use the sidebar to start a new game.")
        return

    snapshot = st.session_state.round_snapshot
    if snapshot:
        render_market_view(snapshot)
    else:
        st.info("No snapshot available. Reset the game from the sidebar.")

    mask = st.session_state.info.get("mask")
    if mask is None:
        mask = np.ones(env.action_space.n, dtype=bool)
    render_action_panel(mask)

    if st.session_state.last_result:
        result = st.session_state.last_result
        pnl = format_money(result["reward"])
        wealth = format_money(result["wealth_after"])
        summary = (
            f"Round {result['round']} | {result['action']} @ {result['exec_price']:.2f} • "
            f"True sum {result['true_sum']:.2f} • PnL {pnl} • Wealth {wealth}"
        )
        st.success(summary)
        reveal_cards = result.get("cards", [])
        if reveal_cards:
            st.markdown("#### Round Reveal")
            render_hint_cards(reveal_cards)
        true_depths = result.get("true_depths")
        if true_depths:
            st.write(
                f"True Liquidity (Bid / Ask): "
                f"{true_depths[0]:.1f} / {true_depths[1]:.1f}"
            )
        st.write(f"Price Impact: {result.get('price_impact', 0.0):+.4f}")

    if st.session_state.done:
        st.warning("Episode finished. Reset from the sidebar to play again.")

    render_history()


if __name__ == "__main__":
    main()
