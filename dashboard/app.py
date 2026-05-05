from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PodVocasem Agent",
    page_icon="🎙️",
    layout="wide",
)

DATA_DIR = Path("data/processed")


@st.cache_data
def load_data():
    with open(DATA_DIR / "episode_metadata.json", encoding="utf-8") as f:
        episodes = json.load(f)
    with open(DATA_DIR / "podcast_dna.json", encoding="utf-8") as f:
        dna = json.load(f)
    with open(DATA_DIR / "topic_graph.json", encoding="utf-8") as f:
        graph = json.load(f)
    with open(DATA_DIR / "similarity_matrix.json", encoding="utf-8") as f:
        similarity = json.load(f)
    stats_path = DATA_DIR / "listening_stats.json"
    stats = json.load(open(stats_path, encoding="utf-8")) if stats_path.exists() else None
    return episodes, dna, graph, similarity, stats


episodes, dna, graph, similarity, stats = load_data()
df = pd.DataFrame(episodes)

# Filtruj jen epizody s existujicim transkriptem
df = df[df["transcript_file"].apply(lambda f: bool(f) and Path(f).exists())].reset_index(drop=True)
episodes = df.to_dict("records")

# ── Session state pro filtry + chat ──────────────────────────────────────────
if "filter_topic" not in st.session_state:
    st.session_state.filter_topic = "Vše"
if "filter_guest" not in st.session_state:
    st.session_state.filter_guest = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Chatbot helpers ───────────────────────────────────────────────────────────
@st.cache_data
def build_transcript_context(question: str = "", max_chars_per_ep: int = 800) -> str:
    """Sestav kontext — hledá relevantní pasáže podle klíčových slov z otázky."""
    # Extrahuj klíčová slova z otázky — použij kořeny slov (prvních 5 znaků)
    # + přidej tematická synonyma pro časté dotazy
    raw_keywords = [w.lower().strip("?!.,") for w in question.split() if len(w) > 3]
    keywords = [w[:5] for w in raw_keywords]  # stem = prvních 5 znaků

    # Tematická rozšíření
    expansions = {
        "knih": ["knih", "knížk", "autor", "přečí", "doporuču", "doporučo", "vydal", "napsal", "napsál", "přečes", "četl", "čteš", "čtení", "sepsal", "dočetl", "dočíst", "přečís", "výborn", "skvělá", "skvělé", "výbor", "titul", "bestse", "vydání", "naklada"],
        "nástr": ["nástr", "tool", "softw", "appli", "platfo", "framew", "knihov"],
        "firmy": ["firmy", "firma", "startu", "společ", "podnik", "založi", "funduji"],
        "přešel": ["přešel", "kariér", "změnil", "přechoz", "začínal", "dřív"],
    }
    expanded = list(keywords)
    for kw in keywords:
        for key, syns in expansions.items():
            if kw.startswith(key[:4]):
                expanded.extend(syns)
    keywords = list(set(expanded))

    parts = []
    for ep in episodes:
        tf = ep.get("transcript_file", "")
        if not tf or not Path(tf).exists():
            continue
        full_text = Path(tf).read_text(encoding="utf-8", errors="ignore")
        header = f"=== EPIZODA: {ep['title']} (host: {ep.get('guest_name','')}, datum: {ep.get('date','')}) ==="

        if keywords:
            # Najdi pasáže kde se vyskytují klíčová slova
            sentences = full_text.replace("\n", " ").split(". ")
            relevant = []
            for i, sent in enumerate(sentences):
                if any(kw in sent.lower() for kw in keywords):
                    # Vezmi kontext: větu před a po
                    chunk = ". ".join(sentences[max(0, i-1):i+2])
                    relevant.append(chunk)
                if len(". ".join(relevant)) > max_chars_per_ep:
                    break

            if relevant:
                parts.append(f"{header}\n{'. '.join(relevant[:5])}")
            else:
                # Žádné klíčové slovo nenalezeno — přidej krátký úvod
                parts.append(f"{header}\n{full_text[:200]}")
        else:
            parts.append(f"{header}\n{full_text[:max_chars_per_ep]}")

    return "\n\n".join(parts)


def ask_claude(question: str, context: str) -> str:
    """Pošli otázku Claudovi s kontextem transkriptů."""
    # Čti API klíč z .env (lokálně) nebo ze Streamlit secrets (cloud)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            pass
    if not api_key:
        return "❌ ANTHROPIC_API_KEY není nastaven."

    client = anthropic.Anthropic(api_key=api_key)
    system = """Jsi AI asistent specializovaný na český IT podcast PodVocasem.
Moderují ho Petr "Poli" Polák a Roman "Džoukr" Provazník.
Máš k dispozici transkripce epizod. Odpovídej v češtině, stručně a konkrétně.
Pokud se ptají na konkrétní informace (knihy, nástroje, firmy), vypiš je jako seznam s uvedením epizody.
Pokud informaci v transkriptech nemáš, řekni to upřímně."""

    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=system,
        messages=[
            {
                "role": "user",
                "content": f"Kontext — transkripce epizod:\n\n{context}\n\n---\n\nOtázka: {question}",
            }
        ],
    )
    return message.content[0].text

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎙️ PodVocasem — AI Agent Dashboard")
st.caption(f"Poli & Džoukr · {len(df)} přepsaných epizod z {dna['total_episodes']} celkem · od {dna['first_episode_date']} do {dna['last_episode_date']}")

# Aktivní filtr badge
if st.session_state.filter_topic != "Vše" or st.session_state.filter_guest:
    col_badge, col_clear = st.columns([4, 1])
    with col_badge:
        parts = []
        if st.session_state.filter_topic != "Vše":
            parts.append(f"Téma: **{st.session_state.filter_topic}**")
        if st.session_state.filter_guest:
            parts.append(f"Host: **{st.session_state.filter_guest}**")
        st.info(f"🔎 Aktivní filtr — {' · '.join(parts)} → přejdi na Episode Explorer")
    with col_clear:
        if st.button("✕ Zrušit filtr"):
            st.session_state.filter_topic = "Vše"
            st.session_state.filter_guest = ""
            st.rerun()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Podcast DNA",
    "🕸️ Knowledge Graph",
    "🔍 Episode Explorer",
    "🌡️ Similarity Heatmap",
    "🤖 Chatbot",
    "📈 Poslechovost",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Podcast DNA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Podcast DNA Profil")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Přepsaných epizod", f"{len(df)} / {dna['total_episodes']}")
    avg_dur = round(df["duration_minutes"][df["duration_minutes"] > 0].mean(), 1)
    col2.metric("Průměrná délka", f"{avg_dur} min")
    col3.metric("Celkem hostů", df["guest_name"].nunique())
    col4.metric("Sezóny", len(dna.get("seasons", [])))

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Rozložení témat")
        st.caption("👆 Klikni na téma → přefiltruje Episode Explorer")
        topic_dist = dna["topic_distribution"]
        topics_filtered = {k: v for k, v in topic_dist.items() if k != "Ostatní"}
        fig_pie = px.pie(
            values=list(topics_filtered.values()),
            names=list(topics_filtered.keys()),
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20))
        pie_event = st.plotly_chart(
            fig_pie, use_container_width=True,
            on_select="rerun", key="pie_chart"
        )
        if pie_event and pie_event.selection and pie_event.selection.points:
            clicked = pie_event.selection.points[0].get("label")
            if clicked and clicked != st.session_state.filter_topic:
                st.session_state.filter_topic = clicked
                st.rerun()

    with col_right:
        st.subheader("Radar — DNA Fingerprint")
        categories = list(topics_filtered.keys())
        values = [v * 100 for v in topics_filtered.values()]
        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.3)",
            line=dict(color="rgb(99, 110, 250)"),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.2])),
            showlegend=False,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Průměrná délka epizody podle tématu")
        dur_by_topic = dna.get("avg_duration_by_topic", {})
        dur_df = pd.DataFrame(
            [(k, v) for k, v in dur_by_topic.items() if k != "Ostatní"],
            columns=["Téma", "Minuty"]
        ).sort_values("Minuty", ascending=True)
        fig_bar = px.bar(
            dur_df, x="Minuty", y="Téma", orientation="h",
            color="Minuty", color_continuous_scale="Blues",
        )
        fig_bar.update_layout(margin=dict(t=20, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.subheader("Epizody v čase")
        df_dates = df[df["date"] != ""].copy()
        df_dates["date"] = pd.to_datetime(df_dates["date"])
        df_dates["year_month"] = df_dates["date"].dt.to_period("Q").astype(str)
        timeline = df_dates.groupby("year_month").size().reset_index(name="count")
        fig_line = px.bar(timeline, x="year_month", y="count", color_discrete_sequence=["#636EFA"])
        fig_line.update_layout(
            xaxis_title="Čtvrtletí", yaxis_title="Počet epizod",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_line, use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Knowledge Graph
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Knowledge Graph — Hosté × Témata × Epizody")
    st.caption("👆 Klikni na hosta → přefiltruje Episode Explorer")

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        show_types = st.multiselect(
            "Zobrazit typy uzlů:",
            ["guest", "topic", "episode"],
            default=["guest", "topic"],
        )
    with col_f2:
        max_nodes = st.slider("Max počet uzlů:", 20, 150, 80)

    G = nx.Graph()
    nodes_data = {n["id"]: n for n in graph["nodes"]}

    added_nodes = set()
    for node in graph["nodes"]:
        if node["type"] in show_types:
            if len(added_nodes) < max_nodes:
                G.add_node(node["id"], **node)
                added_nodes.add(node["id"])

    for edge in graph["edges"]:
        if edge["source"] in added_nodes and edge["target"] in added_nodes:
            G.add_edge(edge["source"], edge["target"], **edge)

    if len(G.nodes) == 0:
        st.warning("Žádné uzly k zobrazení.")
    else:
        pos = nx.spring_layout(G, k=2, seed=42)

        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="#aaa"),
            hoverinfo="none",
        )

        color_map = {"guest": "#636EFA", "topic": "#EF553B", "episode": "#00CC96"}
        size_map = {"guest": 12, "topic": 18, "episode": 8}

        node_traces = []
        all_graph_nodes = []  # pro mapovani klik → node
        trace_offsets = {}
        current_offset = 0

        for ntype in show_types:
            nodes_of_type = [n for n in G.nodes(data=True) if n[1].get("type") == ntype]
            if not nodes_of_type:
                continue
            trace_offsets[ntype] = current_offset
            current_offset += len(nodes_of_type)
            all_graph_nodes.extend(nodes_of_type)

            nx_arr = [pos[n[0]][0] for n in nodes_of_type]
            ny_arr = [pos[n[0]][1] for n in nodes_of_type]
            labels = [n[1].get("label", n[0]) for n in nodes_of_type]
            custom = [n[0] for n in nodes_of_type]  # node ID jako customdata
            hover = [
                f"{n[1].get('label', n[0])}<br>Typ: {ntype}<br>Epizody: {n[1].get('episode_count', '?')}"
                for n in nodes_of_type
            ]
            node_traces.append(go.Scatter(
                x=nx_arr, y=ny_arr,
                mode="markers+text",
                marker=dict(size=size_map[ntype], color=color_map[ntype]),
                text=labels if ntype == "topic" else [],
                textposition="top center",
                hovertext=hover,
                hoverinfo="text",
                customdata=custom,
                name=ntype.capitalize(),
            ))

        fig_graph = go.Figure(data=[edge_trace] + node_traces)
        fig_graph.update_layout(
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        graph_event = st.plotly_chart(
            fig_graph, use_container_width=True,
            on_select="rerun", key="knowledge_graph"
        )
        if graph_event and graph_event.selection and graph_event.selection.points:
            point = graph_event.selection.points[0]
            node_id = point.get("customdata")
            if node_id:
                # Zjisti typ uzlu
                node_info = nodes_data.get(node_id, {})
                ntype = node_info.get("type", "")
                if ntype == "guest":
                    st.session_state.filter_guest = node_id
                    st.session_state.filter_topic = "Vše"
                    st.rerun()
                elif ntype == "topic":
                    st.session_state.filter_topic = node_id
                    st.session_state.filter_guest = ""
                    st.rerun()

    st.divider()
    st.subheader("🔍 Gap Analysis — nepokrytá témata")
    gaps = graph["stats"].get("topic_gaps", [])
    if gaps:
        st.warning(f"Tato témata nejsou v podcastu ještě pokryta: **{', '.join(gaps)}**")
    else:
        st.success("Všechna témata z taxonomie jsou pokryta! 🎉")

    st.subheader("Top 5 nejpokrytějších témat")
    top_topics = graph["stats"].get("most_covered_topics", [])
    if top_topics:
        tt_df = pd.DataFrame(top_topics, columns=["Téma", "Epizod"])
        fig_tt = px.bar(tt_df, x="Téma", y="Epizod", color="Epizod",
                        color_continuous_scale="Viridis")
        fig_tt.update_layout(coloraxis_showscale=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_tt, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Episode Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Episode Explorer")

    col_s1, col_s2, col_s3 = st.columns([3, 2, 2])
    with col_s1:
        search = st.text_input(
            "🔍 Hledat (host, téma, klíčové slovo):",
            value=st.session_state.filter_guest,
        )
    with col_s2:
        all_topics = sorted(set(t for ep in episodes for t in ep.get("topics", [])))
        selected_topic = st.selectbox(
            "Filtrovat podle tématu:",
            ["Vše"] + all_topics,
            index=(["Vše"] + all_topics).index(st.session_state.filter_topic)
            if st.session_state.filter_topic in (["Vše"] + all_topics) else 0,
        )
        # Sync zpet do session state
        st.session_state.filter_topic = selected_topic
    with col_s3:
        min_dur, max_dur = st.slider("Délka (minuty):", 0, 150, (0, 150))

    # Apply filters
    filtered = df.copy()
    if search:
        mask = (
            filtered["title"].str.contains(search, case=False, na=False) |
            filtered["guest_name"].str.contains(search, case=False, na=False) |
            filtered["summary"].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]
    if selected_topic != "Vše":
        filtered = filtered[filtered["topics"].apply(lambda t: selected_topic in t)]
    filtered = filtered[
        (filtered["duration_minutes"] >= min_dur) &
        (filtered["duration_minutes"] <= max_dur)
    ]

    st.caption(f"Zobrazeno {len(filtered)} z {len(df)} epizod")

    display_cols = ["episode_id", "title", "guest_name", "date", "duration_minutes", "topics"]
    display_df = filtered[display_cols].copy()
    display_df.columns = ["#", "Název", "Host", "Datum", "Délka (min)", "Témata"]
    display_df["Témata"] = display_df["Témata"].apply(lambda t: ", ".join(t))
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Detail epizody")
    if len(filtered) > 0:
        selected_ep = st.selectbox("Vyber epizodu:", options=filtered["title"].tolist())
        if selected_ep:
            ep_data = filtered[filtered["title"] == selected_ep].iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Host", ep_data["guest_name"] or "—")
            c2.metric("Délka", f"{ep_data['duration_minutes']} min")
            c3.metric("Datum", ep_data["date"] or "—")
            st.write("**Témata:**", ", ".join(ep_data["topics"]))
            if ep_data["summary"]:
                st.write("**Souhrn:**", ep_data["summary"][:400])
            if ep_data["transcript_file"] and Path(ep_data["transcript_file"]).exists():
                with st.expander("📄 Zobrazit transkript (první 2000 znaků)"):
                    text = Path(ep_data["transcript_file"]).read_text(encoding="utf-8")
                    st.text(text[:2000])
    else:
        st.info("Žádné epizody neodpovídají filtru.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Similarity Heatmap
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Similarity Heatmap — Podobnost epizod")
    st.caption("Čím světlejší políčko, tím více se epizody překrývají tematicky.")

    matrix = similarity["matrix"]
    ep_ids = similarity["episode_ids"]
    ep_titles = [t[:30] for t in similarity["episode_titles"]]

    n_show = st.slider("Počet epizod v heatmapě:", 20, min(87, len(ep_ids)), 40)
    matrix_sub = [row[:n_show] for row in matrix[:n_show]]
    titles_sub = ep_titles[:n_show]

    fig_heat = go.Figure(go.Heatmap(
        z=matrix_sub,
        x=titles_sub,
        y=titles_sub,
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        hovertemplate="<b>%{y}</b><br>vs<br><b>%{x}</b><br>Podobnost: %{z:.0%}<extra></extra>",
    ))
    fig_heat.update_layout(
        height=650,
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        margin=dict(t=20, b=150, l=150),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()
    st.subheader("⚠️ Nejvíce podobné páry epizod")
    st.caption("Tyto epizody se tematicky nejvíce překrývají — pozor na opakování!")
    top_pairs = similarity.get("high_similarity_pairs", [])
    if top_pairs:
        pairs_df = pd.DataFrame(top_pairs[:10])
        pairs_df = pairs_df[["episode_a_title", "episode_b_title", "similarity"]]
        pairs_df.columns = ["Epizoda A", "Epizoda B", "Podobnost"]
        pairs_df["Podobnost"] = pairs_df["Podobnost"].apply(lambda x: f"{x:.0%}")
        st.dataframe(pairs_df, use_container_width=True, hide_index=True)
    else:
        st.success("Žádné výrazně podobné páry nenalezeny. 🎉")

    st.divider()
    avg_sim = similarity["stats"].get("avg_similarity", 0)
    pairs_above = similarity["stats"].get("pairs_above_70pct", 0)
    c1, c2 = st.columns(2)
    c1.metric("Průměrná podobnost", f"{avg_sim:.1%}")
    c2.metric("Páry s >70% překryvem", pairs_above)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Chatbot
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("🤖 Chatbot — Zeptej se na cokoli")
    st.caption(f"Pracuje s {len(df)} přepsanými epizodami · Powered by Claude")

    # Příklady otázek
    st.subheader("Příklady otázek:")
    example_cols = st.columns(3)
    examples = [
        "Jaké knihy hosté doporučili?",
        "Kdo mluvil o Kubernetes?",
        "Jaké startupy hosté zakládali?",
        "Co hosté říkali o AI agentechᵉ?",
        "Kdo přešel z jiného oboru do IT?",
        "Jaké nástroje se nejčastěji zmiňují?",
    ]
    for i, ex in enumerate(examples):
        with example_cols[i % 3]:
            if st.button(ex, key=f"ex_{i}"):
                st.session_state.pending_question = ex

    st.divider()

    # Chat historie
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    question = st.chat_input("Napiš otázku o podcastu...")

    # Zpracuj pending question z tlačítek
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if question:
        # Zobraz otázku
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Získej odpověď
        with st.chat_message("assistant"):
            with st.spinner("Přemýšlím..."):
                context = build_transcript_context(question=question)
                answer = ask_claude(question, context)
            st.write(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Tlačítko pro smazání historie
    if st.session_state.chat_history:
        if st.button("🗑️ Smazat historii chatu"):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Poslechovost
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("📈 Poslechovost — Spotify & Apple Podcasts")

    if not stats:
        st.warning("Data poslechovosti nejsou k dispozici. Spusť nejdřív `pipeline/06_import_stats.py`.")
    else:
        s = stats["summary"]

        # ── Metriky ──────────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Celkem přehrání", f"{s['total_plays']:,}")
        c2.metric("Spotify", f"{s['total_spotify_plays']:,}", f"{s['spotify_share_pct']}%")
        c3.metric("Apple Podcasts", f"{s['total_apple_plays']:,}", f"{s['apple_share_pct']}%")
        c4.metric("Průměr / epizoda", f"{int(s['avg_plays_per_episode']):,}")
        c5.metric("Růst (2. vs 1. polovina)", f"+{s['growth_pct_vs_first_half']}%")

        st.divider()

        # ── Růstový graf ─────────────────────────────────────────────────────
        st.subheader("📊 Denní streamy Spotify — růst podcastu")

        streams_df = pd.DataFrame(stats["spotify_streams_daily"])
        streams_df["date"] = pd.to_datetime(streams_df["date"])
        streams_df = streams_df.set_index("date").resample("W").sum().reset_index()
        streams_df.columns = ["date", "streams"]

        fig_growth = px.area(
            streams_df, x="date", y="streams",
            color_discrete_sequence=["#1DB954"],
            labels={"date": "", "streams": "Streamy / týden"},
        )
        fig_growth.update_layout(margin=dict(t=10, b=10), hovermode="x unified")
        st.plotly_chart(fig_growth, use_container_width=True)

        # Apple měsíční — součet přes epizody
        apple_trends = stats.get("apple_episode_trends", [])
        if apple_trends:
            at_df = pd.DataFrame(apple_trends)
            at_df["date"] = pd.to_datetime(at_df["date"])
            apple_monthly = at_df.groupby("date")["plays"].sum().reset_index()
            apple_monthly.columns = ["date", "plays"]

            st.subheader("🍎 Měsíční přehrání Apple Podcasts — růst podcastu")
            fig_apple = px.area(
                apple_monthly, x="date", y="plays",
                color_discrete_sequence=["#FC3C44"],
                labels={"date": "", "plays": "Přehrání / měsíc"},
            )
            fig_apple.update_layout(margin=dict(t=10, b=10), hovermode="x unified")
            st.plotly_chart(fig_apple, use_container_width=True)

        st.divider()

        # ── Top epizody ───────────────────────────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🏆 Top 15 epizod — celkem")
            eps_df = pd.DataFrame(stats["episodes"])
            eps_df = eps_df[eps_df["episode_code"] != ""].copy()
            top15 = eps_df.nlargest(15, "total_plays")[["title", "spotify_plays", "apple_plays", "total_plays"]].copy()
            top15["title_short"] = top15["title"].apply(lambda t: t[7:47] + "…" if len(t) > 47 else t[7:])

            fig_top = px.bar(
                top15.sort_values("total_plays"),
                x="total_plays", y="title_short",
                orientation="h",
                color_discrete_sequence=["#636EFA"],
                labels={"total_plays": "Přehrání celkem", "title_short": ""},
            )
            fig_top.update_layout(margin=dict(t=10, b=10, l=10), height=450)
            st.plotly_chart(fig_top, use_container_width=True)

        with col_right:
            st.subheader("🎯 Spotify vs Apple — Top 15")
            fig_stacked = px.bar(
                top15.sort_values("total_plays"),
                x=["spotify_plays", "apple_plays"],
                y="title_short",
                orientation="h",
                barmode="stack",
                color_discrete_map={"spotify_plays": "#1DB954", "apple_plays": "#FC3C44"},
                labels={"value": "Přehrání", "title_short": "", "variable": "Platforma"},
            )
            fig_stacked.update_layout(margin=dict(t=10, b=10, l=10), height=450)
            fig_stacked.for_each_trace(lambda t: t.update(
                name="Spotify" if t.name == "spotify_plays" else "Apple"
            ))
            st.plotly_chart(fig_stacked, use_container_width=True)

        st.divider()

        # ── Sezóny ────────────────────────────────────────────────────────────
        st.subheader("📅 Přehrání podle sérií")
        seasons_df = pd.DataFrame(s["seasons"])
        if not seasons_df.empty:
            fig_seasons = px.bar(
                seasons_df, x="season", y="total_plays",
                color="total_plays", color_continuous_scale="Viridis",
                labels={"season": "Série", "total_plays": "Přehrání celkem"},
            )
            fig_seasons.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_seasons, use_container_width=True)

        st.divider()

        # ── Geografie ─────────────────────────────────────────────────────────
        st.subheader("🌍 Odkud poslouchají (Spotify)")
        geo_df = pd.DataFrame(stats["geo"])
        geo_df = geo_df[geo_df["country"] != "unknown"].head(15)
        geo_df["pct_label"] = geo_df["percentage"].apply(lambda x: f"{x:.1f}%")

        col_geo1, col_geo2 = st.columns([2, 1])
        with col_geo1:
            fig_geo = px.bar(
                geo_df, x="percentage", y="country",
                orientation="h",
                color="percentage", color_continuous_scale="Blues",
                labels={"percentage": "% posluchačů", "country": ""},
                text="pct_label",
            )
            fig_geo.update_traces(textposition="outside")
            fig_geo.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10), height=400)
            st.plotly_chart(fig_geo, use_container_width=True)

        with col_geo2:
            st.metric("🇨🇿 Česká republika", f"{geo_df[geo_df['country']=='Czechia']['percentage'].values[0]:.1f}%")
            st.metric("🇸🇰 Slovensko", f"{geo_df[geo_df['country']=='Slovakia']['percentage'].values[0]:.1f}%")
            rest = geo_df[~geo_df["country"].isin(["Czechia", "Slovakia"])]["percentage"].sum()
            st.metric("🌍 Zbytek světa", f"{rest:.1f}%")
            st.caption(f"Celkem {len(stats['geo'])} zemí")
