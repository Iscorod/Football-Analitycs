from __future__ import annotations

import re
import json
import unicodedata
from pathlib import Path

import html as htmllib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
import warnings
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)


ROOT = Path(__file__).resolve().parent
LEAGUE_DIRS = {
    "La Liga": ROOT / "La-Liga_2024-2025_CSV_Raw",
    "Ligue 1": ROOT / "Ligue-1_2024-2025_CSV_Raw",
}
PLAYER_SOURCES = {
    "shooting": "players_shooting",
    "passing": "players_passing",
    "pass_types": "players_passingtypes",
    "creation": "players_gca",
    "goalkeeping": "players_goalkeeping",
    "defense": "players_defense_actions",
    "possession": "players_posession",
    "minutes": "players_playing_time",
    "misc": "players_misc",
}
TEAM_SOURCES = {
    "standard": "squad_standard",
    "shooting": "squad_shooting",
    "passing": "squad_passing",
    "possession": "squad_possession",
    "creation": "squad_goal_shot_creation",
    "defense": "squad_defense",
    "goalkeeping": "squad_goalkeeping",
    "misc": "squad_misc",
}
OPPONENT_SOURCES = {
    "opponent_shooting": "opponent_shooting",
    "opponent_standard": "opponent_standard",
    "opponent_possession": "opponent_possession",
    "opponent_defense": "opponent_defense",
}

ROLE_METRICS = {
    "Delantero": ["Goles/90", "npxG/90", "Tiros a puerta/90", "Acciones de gol/90", "xAG/90"],
    "Extremo": ["Regates con éxito/90", "Centros/90", "xAG/90", "Acciones de gol/90", "npxG/90"],
    "Mediocentro": ["Pases progresivos/90", "Pases clave/90", "Recuperaciones/90", "Tkl+Int/90", "xAG/90"],
    "Central": ["Tkl+Int/90", "Despejes/90", "Duelos aéreos ganados %", "Pases progresivos/90", "Recuperaciones/90"],
    "Lateral": ["Regates con éxito/90", "Pases progresivos/90", "Centros/90", "Tkl+Int/90", "xAG/90"],
    "Portero": ["Paradas %", "Porterías a cero %", "Goles encajados/90 (invertido)"],
}
LOWER_IS_BETTER = {"Goles encajados", "Goles encajados/90"}
STANDARD_SCOUT_PROFILES = {
    "Delantero presionante": {
        "role": "Delantero", "age_max": 24, "min_minutes": 1001,
        "weights": {"Recuperaciones/90": 1.5, "Tackles ganados/90": 1.2, "Intercepciones/90": 0.8, "Acciones de gol/90": 1.0, "npxG/90": 0.8},
    },
    "Mediocentro creador": {
        "role": "Mediocentro", "age_max": 24, "min_minutes": 1001,
        "weights": {"Pases progresivos/90": 1.4, "Pases clave/90": 1.3, "xAG/90": 1.3, "Pases completados %": 0.8, "Asistencias/90": 0.7},
    },
    "Extremo centrador": {
        "role": "Extremo", "age_max": 24, "min_minutes": 1001,
        "weights": {"Centros/90": 1.5, "Pases clave/90": 1.0, "xAG/90": 1.2, "Regates con éxito/90": 0.8, "Acciones de gol/90": 0.8},
    },
    "Defensa aéreo": {
        "role": "Central", "age_max": 24, "min_minutes": 1001,
        "weights": {"Duelos aéreos ganados %": 1.5, "Despejes/90": 1.0, "Tkl+Int/90": 1.0, "Recuperaciones/90": 0.8, "Intercepciones/90": 0.7},
    },
}
TEAM_COMPARE_METRICS = [
    "Pts", "GF", "GA", "GD", "xG equipo", "xGA equipo", "Posesión %", "Tiros/90", "Tiros a puerta %",
    "Pases progresivos/90", "Pases clave/90", "SCA/90", "GCA/90", "Tkl+Int", "Recuperaciones",
]

st.set_page_config(page_title="Scouting | Player Intelligence", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")


def norm(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip().casefold()


def scouting_metric_key(kind: str, metric: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", norm(metric)).strip("_")
    return f"shortlist_{kind}_{slug}"


def load_team_logo_catalog() -> dict[str, dict[str, object]]:
    """Indexa el catálogo por nombre FBref y por todos sus alias normalizados."""
    path = ROOT / "catalogo_escudos.json"
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    index: dict[str, dict[str, object]] = {}
    for entry in catalog.get("teams", {}).values():
        names = [entry.get("fbref_name", ""), *entry.get("aliases", [])]
        for name in names:
            if name:
                index[norm(name)] = entry
    return index


TEAM_LOGOS = load_team_logo_catalog()


def team_name_html(name: object, css_class: str = "team-name-cell") -> str:
    """Nombre de equipo acompañado de su escudo; usa iniciales si no hay URL."""
    team_name = str(name)
    entry = TEAM_LOGOS.get(norm(team_name), {})
    crest_url = entry.get("crest_url")
    safe_name = htmllib.escape(team_name)
    if crest_url:
        mark = f'<img class="team-crest" src="{htmllib.escape(str(crest_url), quote=True)}" alt="" loading="lazy">'
    else:
        initials = "".join(part[0] for part in team_name.split()[:2]).upper() or "⚽"
        mark = f'<span class="team-crest-fallback">{htmllib.escape(initials)}</span>'
    return f'<span class="{css_class}">{mark}<span>{safe_name}</span></span>'


def team_crest_badges(names: list[str], colors: dict[str, str] | None = None) -> None:
    """Muestra los escudos y nombres como leyenda accesible debajo del gráfico."""
    unique_names = list(dict.fromkeys(str(name) for name in names if name and str(name) != "nan"))
    if unique_names:
        badges = "".join(
            f"<span class='team-filter-chip' style='border-color:{htmllib.escape((colors or {}).get(name, 'rgba(255,255,255,.12)'))}'>{team_name_html(name)}</span>"
            for name in unique_names
        )
        st.markdown(f"<div class='team-filter-chips chart-team-legend'>{badges}</div>", unsafe_allow_html=True)


def player_crest_badges(rows: list[pd.Series], colors: dict[str, str] | None = None) -> None:
    """Leyenda visual del gráfico de comparación con jugador y escudo de su club."""
    badges = []
    for row in rows:
        player = htmllib.escape(str(row.get("Jugador", "")))
        club = str(row.get("Equipo", ""))
        team = team_name_html(club, "player-chart-team")
        label = f"{row.get('Jugador', '')} · {club}"
        color = htmllib.escape((colors or {}).get(label, "#b6ff3b"))
        badges.append(f"<span class='player-chart-chip' style='border-color:{color}'><b>{player}</b>{team}</span>")
    if badges:
        st.markdown(f"<div class='team-filter-chips chart-team-legend'>{''.join(badges)}</div>", unsafe_allow_html=True)


def add_team_crests_to_bar_axis(fig: go.Figure, names: list[str]) -> go.Figure:
    """Añade los escudos y etiquetas al eje de un gráfico horizontal de equipos."""
    images = []
    annotations = []
    for name in names:
        entry = TEAM_LOGOS.get(norm(name), {})
        crest_url = entry.get("crest_url")
        if crest_url:
            images.append(dict(
                source=str(crest_url), xref="paper", yref="y", x=0.015, y=name,
                sizex=0.038, sizey=0.68, xanchor="left", yanchor="middle",
                sizing="contain", layer="above",
            ))
        else:
            initials = "".join(part[0] for part in name.split()[:2]).upper() or "⚽"
            annotations.append(dict(
                xref="paper", yref="y", x=0.015, y=name, xanchor="left", yanchor="middle",
                text=f"<b>{htmllib.escape(initials)}</b>", showarrow=False,
                font=dict(size=9, color="#b6ff3b"), bgcolor="rgba(182,255,59,.12)",
                bordercolor="rgba(182,255,59,.26)", borderwidth=1, borderpad=3,
            ))
        annotations.append(dict(
            xref="paper", yref="y", x=0.065, y=name, xanchor="left", yanchor="middle",
            text=htmllib.escape(name), showarrow=False,
            font=dict(size=11, color="#dce6f5"),
        ))
    fig.update_layout(
        images=images, annotations=annotations,
        xaxis=dict(domain=[0.27, 1]),
        yaxis=dict(showticklabels=False),
    )
    return fig


def latest_csv(folder: Path, prefix: str) -> Path | None:
    files = sorted(folder.glob(f"{prefix}_*.csv"))
    return files[-1] if files else None


def csv_signature(folder: Path) -> tuple[tuple[str, int, int], ...]:
    """Cambia la clave de caché cuando aparece, desaparece o se modifica un CSV."""
    if not folder.is_dir():
        return ()
    return tuple(
        (path.name, path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(folder.glob("*.csv"))
    )


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


@st.cache_data(show_spinner=False)
def load_players(folder_text: str, file_signature: tuple[tuple[str, int, int], ...]) -> tuple[pd.DataFrame, list[str]]:
    folder = Path(folder_text)
    warnings: list[str] = []
    base_path = latest_csv(folder, "players_standard")
    if base_path is None:
        base_path = latest_csv(folder, "players_shooting")
        warnings.append("No hay archivo de estadísticas estándar; se usan tiros como tabla principal.")
    if base_path is None:
        return pd.DataFrame(), ["No se encontró ningún CSV de jugadores."]

    base = read_csv(base_path)
    required = ["Player", "Squad"]
    if not all(col in base.columns for col in required):
        return pd.DataFrame(), [f"Faltan las columnas necesarias en {base_path.name}."]
    meta = [c for c in ["Player", "Nation", "Pos", "Squad", "Age", "Born", "MP", "Starts", "Min", "90s"] if c in base.columns]
    data = base[meta].copy()
    data["__player_key"] = data["Player"].map(norm)
    data["__team_key"] = data["Squad"].map(norm)
    data = data.drop_duplicates(["__player_key", "__team_key"], keep="first")

    if base_path.name.startswith("players_standard"):
        base_category = "standard"
    else:
        base_category = "shooting"
    base_stats = base.drop(columns=[c for c in meta if c in base.columns], errors="ignore").copy()
    base_stats["__player_key"] = base["Player"].map(norm)
    base_stats["__team_key"] = base["Squad"].map(norm)
    base_stats = base_stats.drop_duplicates(["__player_key", "__team_key"], keep="first")
    base_stats = base_stats.rename(columns={c: f"{base_category}__{c}" for c in base_stats.columns if not c.startswith("__")})
    data = data.merge(base_stats, on=["__player_key", "__team_key"], how="left")

    for category, prefix in PLAYER_SOURCES.items():
        if category == base_category and base_path.name.startswith(prefix + "_"):
            # The fallback table is already part of the base player rows.
            continue
        path = latest_csv(folder, prefix)
        if path is None:
            warnings.append(f"Falta el archivo {prefix}_*.csv.")
            continue
        src = read_csv(path)
        if not all(col in src.columns for col in required):
            warnings.append(f"{path.name} no contiene Player y Squad.")
            continue
        src["__player_key"] = src["Player"].map(norm)
        src["__team_key"] = src["Squad"].map(norm)
        src = src.drop_duplicates(["__player_key", "__team_key"], keep="first")
        stats = src.drop(columns=[c for c in ["Player", "Nation", "Pos", "Squad", "Age", "Born"] if c in src.columns], errors="ignore")
        stats = stats.rename(columns={c: f"{category}__{c}" for c in stats.columns if not c.startswith("__")})
        data = data.merge(stats, on=["__player_key", "__team_key"], how="left")

    alias = {
        "Goles": ["standard__Gls", "shooting__Gls"],
        "Asistencias": ["standard__Ast", "passing__Ast"],
        "xG": ["standard__xG", "shooting__xG"],
        "npxG": ["standard__npxG", "shooting__npxG"],
        "xAG": ["standard__xAG", "passing__xAG"],
        "Tiros": ["shooting__Sh"],
        "Tiros a puerta": ["shooting__SoT"],
        "Pases clave": ["passing__KP"],
        "Pases progresivos": ["passing__PrgP"],
        "Regates con éxito": ["possession__Succ"],
        "Centros": ["pass_types__Crs"],
        "Tackles ganados": ["defense__TklW"],
        "Tackles": ["defense__Tkl"],
        "Intercepciones": ["defense__Int"],
        "Despejes": ["defense__Clr"],
        "Recuperaciones": ["misc__Recov"],
        "Duelos aéreos ganados %": ["misc__Won%"],
        "Acciones de gol": ["creation__GCA"],
        "Paradas %": ["goalkeeping__Save%"],
        "Porterías a cero %": ["goalkeeping__CS%"],
        "Goles encajados": ["goalkeeping__GA"],
        "Minutos": ["Min", "minutes__Min"],
        "Partidos": ["MP", "minutes__MP"],
        "Titularidades": ["Starts", "minutes__Starts"],
        "90s": ["90s", "minutes__90s"],
        "Pases completados %": ["passing__Cmp%"],
        "xG+xAG": ["standard__npxG+xAG", "standard__xG+xAG"],
    }
    for canonical, candidates in alias.items():
        found = [c for c in candidates if c in data.columns]
        if found:
            data[canonical] = data[found].bfill(axis=1).iloc[:, 0].pipe(numeric)

    if "90s" not in data.columns or data["90s"].isna().all():
        if "Minutos" in data:
            data["90s"] = data["Minutos"] / 90
    data["90s"] = numeric(data.get("90s", pd.Series(index=data.index, dtype=float)))

    for total in ["Goles", "Asistencias", "xG", "npxG", "xAG", "Tiros", "Tiros a puerta", "Pases clave", "Pases progresivos", "Regates con éxito", "Centros", "Tackles ganados", "Intercepciones", "Despejes", "Recuperaciones", "Acciones de gol"]:
        if total in data.columns:
            data[f"{total}/90"] = data[total] / data["90s"].replace(0, pd.NA)
    if "Goles encajados" in data:
        data["Goles encajados/90 (invertido)"] = -(data["Goles encajados"] / data["90s"].replace(0, pd.NA))
    if "xAG" in data:
        data["xAG/90"] = data["xAG"] / data["90s"].replace(0, pd.NA)
    for metric in ["Goles/90", "npxG/90"]:
        if metric not in data:
            continue
    if "Acciones de gol/90" not in data and "Acciones de gol" in data:
        data["Acciones de gol/90"] = data["Acciones de gol"] / data["90s"].replace(0, pd.NA)
    if "Tkl+Int/90" not in data:
        vals = [c for c in ["Tackles", "Intercepciones"] if c in data.columns]
        if vals:
            data["Tkl+Int/90"] = data[vals].sum(axis=1, min_count=1) / data["90s"].replace(0, pd.NA)

    for metric in ["Goles", "Asistencias", "xG", "npxG", "xAG", "Tiros", "Tiros a puerta", "Pases clave", "Pases progresivos", "Regates con éxito", "Centros", "Tackles ganados", "Intercepciones", "Despejes", "Recuperaciones", "Acciones de gol", "Minutos", "Partidos", "Titularidades", "Duelos aéreos ganados %", "Paradas %", "Porterías a cero %", "Pases completados %"]:
        if metric in data.columns:
            data[metric] = numeric(data[metric])
    data["Edad"] = numeric(data["Age"]) if "Age" in data else pd.NA
    if "Pos" in data:
        data["Posición"] = data["Pos"].fillna("").astype(str)
    else:
        data["Posición"] = ""
    data["Jugador"] = data["Player"].astype(str)
    data["Equipo"] = data["Squad"].astype(str)
    data["Nacionalidad"] = data["Nation"].astype(str) if "Nation" in data else ""
    data["Perfil"] = data["Posición"].map(position_label)
    return data, warnings


def position_label(pos: object) -> str:
    val = str(pos).upper()
    if "GK" in val:
        return "Portero"
    if "DF" in val and ("MF" in val or "FW" in val):
        return "Lateral"
    if "DF" in val:
        return "Central"
    if "FW" in val and "MF" in val:
        return "Extremo"
    if "FW" in val:
        return "Delantero"
    if "MF" in val:
        return "Mediocentro"
    return "Sin clasificar"


@st.cache_data(show_spinner=False)
def load_teams(folder_text: str, file_signature: tuple[tuple[str, int, int], ...]) -> tuple[pd.DataFrame, list[str]]:
    folder = Path(folder_text)
    warnings: list[str] = []
    main_path = latest_csv(folder, "overall") or latest_csv(folder, "squad_standard")
    if main_path is None:
        return pd.DataFrame(), ["No se encontraron tablas de equipos."]
    base = read_csv(main_path)
    if "Squad" not in base:
        return pd.DataFrame(), [f"{main_path.name} no contiene Squad."]
    data = base[[c for c in ["Squad", "Rk", "MP", "W", "D", "L", "GF", "GA", "GD", "Pts", "Pts/MP", "xG", "xGA", "xGD", "xGD/90", "Attendance"] if c in base]].copy()
    data["__team_key"] = data["Squad"].map(norm)
    # These CSVs include a synthetic league-average row, not a real club.
    data = data[~data["Squad"].astype(str).isin(["La Liga", "Ligue 1"])].copy()
    if "Rk" in data and "Squad" in base and "Rk" not in base:
        data["Rk"] = pd.NA

    sources = {**TEAM_SOURCES, **OPPONENT_SOURCES}
    for category, prefix in sources.items():
        path = latest_csv(folder, prefix)
        if path is None:
            warnings.append(f"Falta {prefix}_*.csv.")
            continue
        src = read_csv(path)
        if "Squad" not in src:
            continue
        src["__team_key"] = src["Squad"].map(norm)
        drop = [c for c in ["Squad"] if c in src]
        src = src.drop(columns=drop).drop_duplicates("__team_key", keep="first")
        src = src.rename(columns={c: f"{category}__{c}" for c in src.columns if not c.startswith("__")})
        data = data.merge(src, on="__team_key", how="left")

    mapping = {
        "Posesión %": ["possession__Poss", "standard__Poss"],
        "xG equipo": ["shooting__xG", "standard__xG"],
        "xGA equipo": ["standard__xGA"],
        "Tiros/90": ["shooting__Sh/90"],
        "Tiros a puerta %": ["shooting__SoT%"],
        "Pases progresivos": ["passing__PrgP"],
        "Pases clave": ["passing__KP"],
        "SCA/90": ["creation__SCA90"],
        "GCA/90": ["creation__GCA90"],
        "Tkl+Int": ["defense__Tkl+Int"],
        "Recuperaciones": ["misc__Recov"],
        "Goles encajados rival": ["opponent_shooting__Gls", "opponent_standard__Gls"],
        "Tiros recibidos": ["opponent_shooting__Sh"],
        "Posesión rival %": ["opponent_possession__Poss"],
        "Despejes rival": ["opponent_defense__Clr"],
    }
    for canonical, candidates in mapping.items():
        found = [c for c in candidates if c in data]
        if found:
            data[canonical] = data[found].bfill(axis=1).iloc[:, 0].pipe(numeric)
    team_90s = numeric(data["standard__90s"]) if "standard__90s" in data else numeric(data["MP"]) if "MP" in data else pd.Series(1.0, index=data.index)
    for total, rate in [("Pases progresivos", "Pases progresivos/90"), ("Pases clave", "Pases clave/90")]:
        if total in data:
            data[rate] = data[total] / team_90s.replace(0, pd.NA)
    for col in data.columns:
        if col != "Squad" and not col.startswith("__"):
            converted = numeric(data[col])
            if converted.notna().any():
                data[col] = converted
    for metric in ["xG equipo", "xGA equipo", "Posesión %", "Tiros/90", "Tiros a puerta %", "Pases progresivos/90", "Pases clave/90", "SCA/90", "GCA/90", "Tkl+Int"]:
        if metric in data and data[metric].notna().any():
            data[f"pct__{metric}"] = data[metric].rank(pct=True) * 100
    return data, warnings


def percentile_score(df: pd.DataFrame, metrics: list[str]) -> pd.Series:
    values = []
    for metric in metrics:
        if metric in df and numeric(df[metric]).notna().sum() > 1:
            vals = numeric(df[metric])
            if metric == "Paradas %" or metric == "Porterías a cero %" or metric == "Duelos aéreos ganados %":
                pass
            values.append(vals.rank(pct=True) * 100)
    if not values:
        return pd.Series(pd.NA, index=df.index)
    return pd.concat(values, axis=1).mean(axis=1)


def weighted_profile_scores(
    peers: pd.DataFrame,
    candidates: pd.DataFrame,
    metrics: list[str],
    weights: dict[str, float],
    higher_is_better: dict[str, bool],
) -> list[float | None]:
    """Calcula índices ponderados contra percentiles del grupo de referencia."""
    totals: dict[tuple[str, str], list[float]] = {}
    for metric in metrics:
        weight = float(weights.get(metric, 0.0))
        if weight <= 0 or metric not in peers:
            continue
        values = numeric(peers[metric])
        percentile = values.rank(pct=True, ascending=higher_is_better.get(metric, True))
        for index, peer in peers.iterrows():
            value = percentile.loc[index]
            if pd.isna(value):
                continue
            key = (str(peer["__player_key"]), str(peer["__team_key"]))
            acc = totals.setdefault(key, [0.0, 0.0])
            acc[0] += float(value) * 100 * weight
            acc[1] += weight
    scores = []
    for _, candidate in candidates.iterrows():
        key = (str(candidate["__player_key"]), str(candidate["__team_key"]))
        total, weight = totals.get(key, [0.0, 0.0])
        scores.append(total / weight if weight else None)
    return scores


def position_filter(df: pd.DataFrame, role: str) -> pd.DataFrame:
    pos = df["Posición"].fillna("").astype(str).str.upper()
    masks = {
        "Delantero": pos.str.contains("FW") & ~pos.str.contains("MF") & ~pos.str.contains("DF"),
        "Extremo": pos.str.contains("FW") & pos.str.contains("MF") & ~pos.str.contains("DF"),
        "Mediocentro": pos.str.contains("MF") & ~pos.str.contains("DF") & ~pos.str.contains("FW"),
        "Central": pos.str.contains("DF") & ~pos.str.contains("MF") & ~pos.str.contains("FW"),
        "Lateral": pos.str.contains("DF") & (pos.str.contains("MF") | pos.str.contains("FW")),
        "Portero": pos.str.contains("GK"),
    }
    return df[masks.get(role, pd.Series(True, index=df.index))]


def player_name_options(df: pd.DataFrame) -> list[str]:
    return (df["Jugador"] + " · " + df["Equipo"]).tolist()


def metric_value_table(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric in df.columns and numeric(df[metric]).notna().any():
            vals = numeric(df[metric])
            if metric.endswith("%"):
                label = metric
            elif metric in ["Goles encajados/90 (invertido)"]:
                label = "Goles encajados/90"
                vals = -vals
            else:
                label = metric
            rows.append({"Métrica": label, **{str(i + 1): v for i, v in enumerate(vals)}})
    return pd.DataFrame(rows)



# ══════════════════════════════════════════════════════════════════════════════
#  UI · tema oscuro, animaciones y gráficos con entrada animada
# ══════════════════════════════════════════════════════════════════════════════
pio.templates.default = "plotly_dark"
LIME, CYAN, PINK = "#b6ff3b", "#38e1ff", "#ff4d8d"
PALETTE = [LIME, CYAN, PINK, "#8b5cf6", "#ffb84d", "#4ade80", "#60a5fa", "#f472b6", "#fb923c", "#a78bfa", "#2dd4bf", "#facc15"]
LIME_SCALE = ["#14301f", "#4d8a1f", LIME]
GRID = "rgba(255,255,255,.07)"
TEXT_SECONDARY = "#c2cde0"
TEXT_MUTED = "#b4c0d3"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
@property --n { syntax: '<integer>'; initial-value: 0; inherits: false; }
@property --p { syntax: '<number>'; initial-value: 0; inherits: false; }
:root { --lime:#b6ff3b; --cyan:#38e1ff; --pink:#ff4d8d; --ink:#e8eefb; --mute:#b4c0d3; --line:rgba(255,255,255,.14); }

/* ── Fondo: aurora animada + rejilla ───────────────────────────── */
.stApp { background:#05070d; color:var(--ink); font-family:'Inter',sans-serif; }
.stApp::before { content:""; position:fixed; inset:-20%; z-index:0; pointer-events:none;
  background:
    radial-gradient(520px 380px at 18% 14%, rgba(182,255,59,.16), transparent 70%),
    radial-gradient(620px 440px at 82% 22%, rgba(56,225,255,.15), transparent 70%),
    radial-gradient(560px 460px at 55% 92%, rgba(255,77,141,.13), transparent 70%);
  animation:drift 22s ease-in-out infinite alternate; }
@keyframes drift { from { transform:translate3d(-2%,-1%,0) scale(1); } to { transform:translate3d(3%,2%,0) scale(1.12); } }
.stApp::after { content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
  background-size:56px 56px;
  -webkit-mask-image:radial-gradient(ellipse at 50% 0%, #000 0%, transparent 75%);
  mask-image:radial-gradient(ellipse at 50% 0%, #000 0%, transparent 75%); }
[data-testid="stMain"] { position:relative; z-index:1; }
[data-testid="stHeader"] { background:transparent !important; }
.block-container { padding-top:1.3rem !important; max-width:1280px; }
::-webkit-scrollbar { width:8px; height:8px; }
::-webkit-scrollbar-thumb { background:rgba(255,255,255,.14); border-radius:8px; }

@keyframes rise { from { opacity:0; transform:translateY(22px); } to { opacity:1; transform:none; } }
[data-testid="stMain"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"],
[data-testid="stMain"] [data-testid="stVerticalBlock"] > .element-container { animation:rise .7s cubic-bezier(.2,.8,.2,1) backwards; }
[data-testid="stElementContainer"]:nth-child(2) { animation-delay:.06s !important; }
[data-testid="stElementContainer"]:nth-child(3) { animation-delay:.12s !important; }
[data-testid="stElementContainer"]:nth-child(4) { animation-delay:.18s !important; }
[data-testid="stElementContainer"]:nth-child(5) { animation-delay:.24s !important; }

/* ── Hero ──────────────────────────────────────────────────────── */
.hero { position:relative; overflow:hidden; padding:.9rem 1.55rem; border-radius:22px; border:1px solid var(--line); margin:.1rem 0 .8rem;
  background:linear-gradient(120deg, rgba(15,28,46,.88), rgba(10,16,30,.7)); backdrop-filter:blur(14px);
  box-shadow:0 30px 80px -30px rgba(0,0,0,.85), inset 0 1px 0 rgba(255,255,255,.08); animation:rise .8s cubic-bezier(.2,.8,.2,1) backwards; }
.hero::before { content:""; position:absolute; inset:0; pointer-events:none;
  background:linear-gradient(110deg, transparent 30%, rgba(182,255,59,.10) 50%, transparent 70%); transform:translateX(-100%); animation:sheen 5.5s ease-in-out 1.2s infinite; }
@keyframes sheen { to { transform:translateX(100%); } }
.badge { display:inline-flex; align-items:center; gap:.48rem; padding:.27rem .72rem; border-radius:999px; background:rgba(182,255,59,.1);
  border:1px solid rgba(182,255,59,.35); color:var(--lime); font-size:.66rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; }
.badge i { width:7px; height:7px; border-radius:50%; background:var(--lime); animation:pulse 1.8s infinite; }
@keyframes pulse { 0% { box-shadow:0 0 0 0 rgba(182,255,59,.7); } 70% { box-shadow:0 0 0 10px rgba(182,255,59,0); } 100% { box-shadow:0 0 0 0 rgba(182,255,59,0); } }
.hero h1 { font-family:'Space Grotesk',sans-serif; font-size:clamp(1.7rem,3.1vw,2.45rem) !important; line-height:1.02 !important; margin:.35rem 0 .28rem !important;
  padding:0 !important; font-weight:700 !important; letter-spacing:-.03em; color:#fff; position:relative; }
.grad { background:linear-gradient(90deg, var(--lime), var(--cyan), var(--pink), var(--lime)); background-size:300% 100%;
  -webkit-background-clip:text; background-clip:text; color:transparent; animation:flow 7s linear infinite; }
@keyframes flow { to { background-position:300% 0; } }
.hero p { max-width:570px; color:#a9b6cc; font-size:.86rem; line-height:1.45; margin:0 0 .5rem; position:relative; }
.chips { display:flex; flex-wrap:wrap; gap:.32rem; position:relative; }
.chips span { padding:.16rem .55rem; border-radius:999px; font-size:.68rem; color:#cfe0f5; background:rgba(255,255,255,.06); border:1px solid var(--line); }
.pitch { position:absolute; right:1.45rem; top:50%; width:min(31%,320px); color:var(--lime); opacity:.36;
  transform:translateY(-50%) perspective(700px) rotateX(26deg) rotateZ(-6deg); filter:drop-shadow(0 0 8px rgba(182,255,59,.65)); }
.pitch * { stroke-dasharray:1; stroke-dashoffset:1; animation:draw 2.8s cubic-bezier(.6,0,.2,1) .3s forwards; }
@keyframes draw { to { stroke-dashoffset:0; } }
@media (max-width:900px) { .pitch { display:none; } .hero { padding:.8rem 1.15rem; } .hero p { max-width:720px; } }

/* ── Cabeceras de sección ─────────────────────────────────────── */
.sec { display:flex; align-items:baseline; flex-wrap:wrap; gap:.3rem .9rem; margin:.5rem 0 .9rem; }
.sec-t { font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:700; color:#fff; letter-spacing:-.02em; }
.sec-t::before { content:""; display:inline-block; width:6px; height:1.05em; margin-right:.6rem; vertical-align:-.14em; border-radius:3px; background:linear-gradient(var(--lime), var(--cyan)); }
.sec-s { color:var(--mute); font-size:.9rem; }

/* ── KPIs con contador animado ────────────────────────────────── */
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(190px,1fr)); gap:1rem; margin:.3rem 0 1.3rem; }
.kpi { position:relative; overflow:hidden; padding:1.1rem 1.3rem 1.2rem; border-radius:20px; border:1px solid var(--line);
  background:linear-gradient(150deg, rgba(255,255,255,.075), rgba(255,255,255,.02)); backdrop-filter:blur(10px);
  transition:transform .35s cubic-bezier(.2,.8,.2,1), border-color .35s, box-shadow .35s; animation:rise .7s cubic-bezier(.2,.8,.2,1) backwards; }
.kpi:hover { transform:translateY(-5px); border-color:rgba(182,255,59,.5); box-shadow:0 18px 40px -18px rgba(182,255,59,.45); }
.kpi::after { content:""; position:absolute; left:0; top:0; width:100%; height:3px; background:linear-gradient(90deg, var(--lime), var(--cyan), transparent); }
.kpi-ico { font-size:1.25rem; }
.kpi-lab { color:var(--mute); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; font-weight:600; margin-top:.35rem; }
.kpi-val { font-family:'Space Grotesk',sans-serif; font-size:2.35rem; font-weight:700; color:#fff; line-height:1.1; margin-top:.1rem; }
.compare-score-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:1rem; margin:.35rem 0 1.3rem; }
.compare-score-card { position:relative; overflow:hidden; min-height:128px; padding:1rem 1.15rem; border-radius:20px; border:1px solid color-mix(in srgb,var(--score-color) 52%,transparent); background:linear-gradient(150deg,rgba(255,255,255,.075),rgba(255,255,255,.02)); backdrop-filter:blur(10px); transition:transform .3s,border-color .3s,box-shadow .3s; }
.compare-score-card::after { content:""; position:absolute; left:0; top:0; width:100%; height:3px; background:linear-gradient(90deg,var(--score-color),transparent); }
.compare-score-card:hover { transform:translateY(-3px); border-color:var(--score-color); box-shadow:0 16px 34px -20px var(--score-color); }
.compare-score-label { color:#c2cde0; font-size:.76rem; letter-spacing:.08em; text-transform:uppercase; }
.compare-score-value { margin:.08rem 0 .2rem; font-family:'Space Grotesk',sans-serif; font-size:2.35rem; font-weight:700; line-height:1.1; color:var(--score-color); }
.compare-score-club { color:#b4c0d3; font-size:.76rem; }
.count { --n:0; counter-reset:n var(--n); animation:count 1.7s cubic-bezier(.2,.8,.2,1) .2s backwards; }
.count::after { content:counter(n); }
@keyframes count { from { --n:0; } }

/* ── Ficha de jugador ─────────────────────────────────────────── */
.pcard { display:grid; grid-template-columns:auto 1fr; gap:1.8rem; align-items:center; padding:1.6rem 1.9rem; margin-bottom:1.2rem; border-radius:28px;
  border:1px solid var(--line); backdrop-filter:blur(12px); animation:rise .7s cubic-bezier(.2,.8,.2,1) backwards;
  background:radial-gradient(520px 230px at 0% 0%, rgba(182,255,59,.14), transparent 70%), linear-gradient(150deg, rgba(255,255,255,.07), rgba(255,255,255,.02)); }
.ring { position:relative; width:134px; height:134px; border-radius:50%; display:flex; flex-direction:column; align-items:center; justify-content:center;
  background:conic-gradient(var(--lime) calc(var(--p) * 1%), rgba(255,255,255,.09) 0); animation:fill 1.9s cubic-bezier(.2,.8,.2,1) backwards;
  filter:drop-shadow(0 0 14px rgba(182,255,59,.35)); }
.ring::before { content:""; position:absolute; inset:10px; border-radius:50%; background:#0a1020; }
.ring > * { position:relative; z-index:1; }
.ring b { font-family:'Space Grotesk',sans-serif; font-size:2.5rem; color:#fff; line-height:1; }
.ring small { color:var(--mute); font-size:.62rem; letter-spacing:.16em; text-transform:uppercase; margin-top:.2rem; }
@keyframes fill { from { --p:0; } }
.pname { font-family:'Space Grotesk',sans-serif; font-size:clamp(1.6rem,3vw,2.3rem); font-weight:700; color:#fff; letter-spacing:-.02em; line-height:1.1; }
.pteam { color:var(--lime); font-weight:600; margin:.2rem 0 .6rem; }
.tag { display:inline-block; margin:0 .4rem .3rem 0; padding:.2rem .7rem; border-radius:999px; font-size:.75rem; color:#cfe0f5; background:rgba(255,255,255,.06); border:1px solid var(--line); }
.pstats { display:grid; grid-template-columns:repeat(5, 1fr); gap:.7rem; margin-top:.8rem; }
.pstat { padding:.65rem .85rem; border-radius:14px; background:rgba(255,255,255,.05); border:1px solid var(--line); transition:transform .3s, border-color .3s; }
.pstat:hover { transform:translateY(-3px); border-color:rgba(182,255,59,.5); }
.pstat b { display:block; font-family:'Space Grotesk',sans-serif; font-size:1.4rem; color:#fff; }
.pstat span { font-size:.64rem; letter-spacing:.12em; text-transform:uppercase; color:var(--mute); }
@media (max-width:760px) { .pcard { grid-template-columns:1fr; justify-items:center; text-align:center; } .pstats { grid-template-columns:repeat(3,1fr); } }

/* ── Tabs tipo "pill" ─────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap:4px; padding:6px; width:fit-content; max-width:100%; border-radius:999px; background:rgba(255,255,255,.045);
  border:1px solid var(--line); backdrop-filter:blur(12px); box-shadow:0 10px 30px -15px rgba(0,0,0,.8); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none !important; }
.stTabs button[data-baseweb="tab"] { height:auto; padding:.55rem 1.15rem; border-radius:999px; color:var(--mute); transition:all .3s cubic-bezier(.2,.8,.2,1); }
.stTabs button[data-baseweb="tab"] p { font-weight:600; font-size:.92rem; }
.stTabs button[data-baseweb="tab"]:hover { color:#fff; background:rgba(255,255,255,.06); }
.stTabs button[aria-selected="true"] { background:linear-gradient(90deg, var(--lime), #7dff7a); box-shadow:0 8px 24px -6px rgba(182,255,59,.6); }
.stTabs button[aria-selected="true"] p { color:#07110a !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:1.4rem; animation:panelIn .6s cubic-bezier(.2,.8,.2,1) backwards; }
@keyframes panelIn { from { opacity:0; transform:translateY(24px) scale(.99); filter:blur(6px); } to { opacity:1; transform:none; filter:none; } }

/* ── Componentes nativos ──────────────────────────────────────── */
div[data-testid="stMetric"] { position:relative; overflow:hidden; padding:1rem 1.15rem; border-radius:18px; border:1px solid var(--line);
  background:linear-gradient(150deg, rgba(255,255,255,.07), rgba(255,255,255,.02)); transition:transform .3s, border-color .3s, box-shadow .3s; }
div[data-testid="stMetric"]:hover { transform:translateY(-4px); border-color:rgba(182,255,59,.5); box-shadow:0 16px 36px -18px rgba(182,255,59,.45); }
div[data-testid="stMetricLabel"] p { color:var(--mute); text-transform:uppercase; letter-spacing:.1em; font-size:.72rem; font-weight:600; }
div[data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif; font-weight:700; color:#fff; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div { background:rgba(255,255,255,.05) !important; border:1px solid var(--line) !important; border-radius:12px !important; transition:border-color .25s; }
[data-baseweb="select"] > div:hover { border-color:rgba(182,255,59,.55) !important; }
[data-baseweb="tag"] { background:rgba(182,255,59,.16) !important; border-radius:8px !important; }
[data-baseweb="tag"] span { color:var(--lime) !important; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:16px; overflow:hidden; box-shadow:0 18px 40px -24px rgba(0,0,0,.9); }
.stApp [data-testid="stDataFrame"] { background:#0d1424 !important; border:1px solid rgba(182,255,59,.22); }
[data-testid="stExpander"] { border:1px solid rgba(182,255,59,.2) !important; border-radius:16px !important; background:rgba(13,20,36,.92) !important; overflow:hidden; }
[data-testid="stExpander"] summary { background:rgba(13,20,36,.92) !important; color:#dce6f5 !important; }
[data-testid="stExpander"] summary p { color:#dce6f5 !important; }
[data-testid="stExpander"] summary:hover { background:rgba(182,255,59,.07) !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { background:rgba(13,20,36,.92) !important; }
[data-testid="stAlert"] { border-radius:14px; }
[data-testid="stCaptionContainer"] { color:var(--mute); }
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
[data-testid="stRadio"] label, [data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label, [data-testid="stSlider"] label,
[data-testid="stExpander"] summary, [data-testid="stCaptionContainer"] {
  color:#c2cde0;
}
[data-testid="stMarkdownContainer"] small, [data-testid="stCaptionContainer"] p {
  color:#b4c0d3 !important;
}
.stApp [data-testid="stDataFrame"] {
  background:#0d1424 !important;
  border:1px solid rgba(182,255,59,.22);
}
.stDownloadButton button, .stButton button { border-radius:999px !important; border:0 !important; padding:.55rem 1.4rem !important;
  background:linear-gradient(90deg, var(--lime), #7dff7a) !important; transition:transform .25s, box-shadow .25s; }
.stDownloadButton button p, .stButton button p { color:#07110a !important; font-weight:700; }
.stDownloadButton button:hover, .stButton button:hover { transform:translateY(-2px); box-shadow:0 12px 30px -8px rgba(182,255,59,.65); }
iframe { border-radius:18px; }

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] { background:rgba(7,11,20,.85) !important; backdrop-filter:blur(16px); border-right:1px solid var(--line); }
.brand { display:flex; align-items:center; gap:.8rem; margin:.3rem 0 1.4rem; }
.brand-logo { width:44px; height:44px; display:grid; place-items:center; font-size:1.4rem; border-radius:14px;
  background:linear-gradient(135deg, var(--lime), var(--cyan)); box-shadow:0 8px 24px -6px rgba(182,255,59,.6); animation:float 3.5s ease-in-out infinite; }
@keyframes float { 50% { transform:translateY(-4px) rotate(-6deg); } }
.brand-t { font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:700; color:#fff; line-height:1.1; }
.brand-t em { color:var(--lime); font-style:normal; }
.brand-s { color:var(--mute); font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; }
.side-h { color:var(--mute); font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; font-weight:700; margin-bottom:.4rem; }
.foot { text-align:center; color:#b4c0d3; font-size:.78rem; margin:2.5rem 0 .5rem; }
.foot b { color:var(--lime); font-weight:600; }
.standings-wrap { overflow-x:auto; border:1px solid rgba(182,255,59,.2); border-radius:16px; background:rgba(13,20,36,.92); box-shadow:0 18px 40px -24px rgba(0,0,0,.9); }
.standings { width:100%; border-collapse:collapse; color:#e8eefb; font-size:.9rem; font-variant-numeric:tabular-nums; }
.standings thead { background:linear-gradient(90deg,rgba(182,255,59,.13),rgba(56,225,255,.07)); }
.standings th { padding:.85rem 1rem; text-align:right; color:#c2cde0; font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; border-bottom:1px solid rgba(255,255,255,.14); }
.standings th:nth-child(2), .standings td.club { text-align:left; }
.standings td { padding:.8rem 1rem; text-align:right; border-bottom:1px solid rgba(255,255,255,.07); color:#cbd6e7; white-space:nowrap; }
.standings tbody tr { background:rgba(255,255,255,calc(var(--row) * .025)); transition:background .18s ease; }
.standings tbody tr:hover { background:rgba(182,255,59,.09); }
.standings td.rank { color:#b6ff3b; font-weight:700; }
.standings td.club { color:#f4f7fb; font-weight:650; }
.team-name-cell { display:inline-flex; align-items:center; gap:.62rem; min-width:0; }
.team-crest, .team-crest-fallback { width:25px; height:25px; flex:0 0 25px; object-fit:contain; }
.team-crest-fallback { display:inline-grid; place-items:center; border-radius:50%; color:#b6ff3b; background:rgba(182,255,59,.12); border:1px solid rgba(182,255,59,.26); font-size:.62rem; font-weight:800; }
.shortlist-table .team-crest, .shortlist-table .team-crest-fallback { width:23px; height:23px; flex-basis:23px; }
.team-select-card { display:flex; align-items:center; gap:.85rem; margin:.3rem 0 1rem; padding:.75rem 1rem; width:fit-content; border:1px solid rgba(182,255,59,.18); border-radius:14px; background:rgba(13,20,36,.72); color:#f4f7fb; font-weight:650; }
.team-select-card .team-crest, .team-select-card .team-crest-fallback { width:34px; height:34px; flex-basis:34px; }
.team-filter-chips { display:flex; flex-wrap:wrap; gap:.5rem; margin:.25rem 0 .75rem; }
.chart-team-legend { margin:.65rem 0 1rem; }
.team-filter-chip { display:inline-flex; align-items:center; gap:.42rem; padding:.28rem .6rem; border:1px solid rgba(255,255,255,.12); border-radius:999px; background:rgba(255,255,255,.045); color:#e8eefb; font-size:.78rem; }
.team-filter-chip .team-crest, .team-filter-chip .team-crest-fallback { width:19px; height:19px; flex-basis:19px; }
.player-chart-chip { display:inline-flex; align-items:center; gap:.48rem; padding:.32rem .62rem; border:1px solid rgba(255,255,255,.12); border-radius:999px; background:rgba(255,255,255,.045); color:#e8eefb; font-size:.77rem; }
.player-chart-team { display:inline-flex; align-items:center; gap:.32rem; color:#c2cde0; font-weight:500; }
.player-chart-team .team-crest, .player-chart-team .team-crest-fallback { width:19px; height:19px; flex-basis:19px; }
.player-team-name { color:inherit; }
.shortlist-wrap { max-height:590px; overflow:auto; }
.shortlist-table { min-width:1050px; }
.shortlist-table th, .shortlist-table td { text-align:right; }
.shortlist-table th:first-child, .shortlist-table td:first-child,
.shortlist-table th:nth-child(2), .shortlist-table td:nth-child(2),
.shortlist-table th:nth-child(3), .shortlist-table td:nth-child(3) { text-align:left; }
.shortlist-table thead { position:sticky; top:0; z-index:2; }
.shortlist-table td.score { color:#b6ff3b; font-weight:750; }
.shortlist-table td.team { color:#b4c0d3; }
.compare-wrap { overflow-x:auto; border:1px solid rgba(182,255,59,.2); border-radius:16px; background:rgba(13,20,36,.92); box-shadow:0 18px 40px -24px rgba(0,0,0,.9); }
.compare-table { width:100%; min-width:660px; border-collapse:collapse; color:#e8eefb; font-size:.88rem; font-variant-numeric:tabular-nums; }
.compare-table thead { background:linear-gradient(90deg,rgba(182,255,59,.13),rgba(56,225,255,.07)); }
.compare-table th { min-width:190px; padding:.85rem 1rem; text-align:left; vertical-align:middle; color:#f4f7fb; border-bottom:1px solid rgba(255,255,255,.14); }
.compare-table th.compare-metric-head { min-width:150px; color:#b6ff3b; }
.compare-player-head { display:flex; flex-direction:column; align-items:flex-start; gap:.38rem; }
.compare-player-head > b { font-size:.9rem; }
.compare-team { display:inline-flex; align-items:center; gap:.4rem; color:#c2cde0; font-size:.76rem; font-weight:500; }
.compare-team .team-crest, .compare-team .team-crest-fallback { width:21px; height:21px; flex-basis:21px; }
.compare-table td { padding:.72rem 1rem; border-bottom:1px solid rgba(255,255,255,.07); color:#d5deec; }
.compare-table tbody tr { background:rgba(255,255,255,calc(var(--row) * .025)); transition:background .18s ease; }
.compare-table tbody tr:hover { background:rgba(182,255,59,.08); }
.compare-table tbody tr:last-child td { border-bottom:0; }
.compare-table td.compare-metric { color:#f4f7fb; font-weight:600; }
.compare-table td.compare-value { text-align:right; }
.player-metrics-wrap { max-height:420px; overflow:auto; border:1px solid rgba(182,255,59,.2); border-radius:14px; background:rgba(13,20,36,.92); }
.player-metrics-table { width:100%; border-collapse:collapse; color:#e8eefb; font-size:.86rem; font-variant-numeric:tabular-nums; }
.player-metrics-table thead { position:sticky; top:0; z-index:1; background:linear-gradient(90deg,rgba(182,255,59,.15),rgba(56,225,255,.08)); }
.player-metrics-table th { padding:.68rem 1rem; text-align:left; color:#c2cde0; font-size:.7rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; border-bottom:1px solid rgba(255,255,255,.14); }
.player-metrics-table th:last-child, .player-metrics-table td:last-child { width:140px; text-align:right; }
.player-metrics-table td { padding:.55rem 1rem; border-bottom:1px solid rgba(255,255,255,.07); color:#d5deec; }
.player-metrics-table tbody tr { background:rgba(255,255,255,calc(var(--row) * .025)); transition:background .18s ease; }
.player-metrics-table tbody tr:hover { background:rgba(182,255,59,.08); }
.player-metrics-table tbody tr:last-child td { border-bottom:0; }
.profile-metric-label { display:flex; align-items:center; min-height:2.5rem; color:#e8eefb; font-size:.88rem; font-weight:600; }
.standings tbody tr:last-child td { border-bottom:0; }
@media (prefers-reduced-motion:reduce) { *, *::before, *::after { animation-duration:.01ms !important; animation-delay:0s !important; } }
</style>
"""

PITCH_SVG = (
    '<svg class="pitch" viewBox="0 0 400 260" fill="none" stroke="currentColor" stroke-width="1.6">'
    '<rect pathLength="1" x="2" y="2" width="396" height="256" rx="6"/>'
    '<line pathLength="1" x1="200" y1="2" x2="200" y2="258"/>'
    '<circle pathLength="1" cx="200" cy="130" r="36"/>'
    '<rect pathLength="1" x="2" y="70" width="62" height="120"/>'
    '<rect pathLength="1" x="336" y="70" width="62" height="120"/>'
    '<rect pathLength="1" x="2" y="102" width="22" height="56"/>'
    '<rect pathLength="1" x="376" y="102" width="22" height="56"/>'
    '<circle pathLength="1" cx="200" cy="130" r="3"/>'
    "</svg>"
)

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="hero">' + PITCH_SVG +
    '<span class="badge"><i></i>Recruitment intelligence · Temporada 2024–25</span>'
    '<h1>Scouting <span class="grad">Intelligence</span></h1>'
    "<p>Filtra, compara y contextualiza el rendimiento de jugadores y equipos de La Liga y Ligue 1 con métricas por 90 minutos, xG y percentiles por perfil.</p>"
    '<div class="chips"><span>⚽ La Liga</span><span>🇫🇷 Ligue 1</span><span>📈 xG · xAG</span><span>🎯 Índice de perfil</span><span>🆚 Comparador</span></div>'
    "</div>",
    unsafe_allow_html=True,
)


# ───────────────────────────── helpers de UI ─────────────────────────────
def sec(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="sec"><div class="sec-t">{htmllib.escape(title)}</div><div class="sec-s">{htmllib.escape(sub)}</div></div>', unsafe_allow_html=True)


def kpi_cards(items: list[tuple[str, str, object]]) -> None:
    cards = ""
    for i, (icon, label, value) in enumerate(items):
        if isinstance(value, (int, float, np.integer, np.floating)) and pd.notna(value):
            val = f'<span class="count" style="--n:{int(round(float(value)))}"></span>'
        else:
            val = htmllib.escape(str(value))
        cards += f'<div class="kpi" style="animation-delay:{i * 90}ms"><div class="kpi-ico">{icon}</div><div class="kpi-lab">{htmllib.escape(label)}</div><div class="kpi-val">{val}</div></div>'
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def table(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def standings_table(df: pd.DataFrame) -> None:
    """Tabla de clasificación con tema oscuro, contraste alto y columnas alineadas."""
    columns = [c for c in ["Rk", "Squad", "MP", "W", "D", "L", "GF", "GA", "GD", "Pts", "xG", "xGA", "xGD"] if c in df]
    if not columns:
        st.info("No hay columnas de clasificación disponibles.")
        return
    headers = "".join(f"<th>{htmllib.escape(c)}</th>" for c in columns)
    ranked = df.sort_values("Rk", ascending=True) if "Rk" in df else df.sort_values("Pts", ascending=False)
    rows = []
    for place, (_, row) in enumerate(ranked.iterrows(), start=1):
        cells = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                rendered = "—"
            elif col == "Squad":
                rendered = team_name_html(value)
            elif col in {"Rk", "MP", "W", "D", "L", "GF", "GA", "Pts"}:
                rendered = f"{float(value):,.0f}"
            else:
                rendered = f"{float(value):,.2f}"
            cls = "club" if col == "Squad" else ("rank" if col == "Rk" else "num")
            cells.append(f'<td class="{cls}">{rendered}</td>')
        rows.append(f'<tr style="--row:{place % 2}">{"".join(cells)}</tr>')
    st.markdown(
        "<div class='standings-wrap'><table class='standings'><thead><tr>" + headers + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def shortlist_table(df: pd.DataFrame) -> None:
    """Tabla de candidatos con el mismo lenguaje visual que la clasificación."""
    columns = list(df.columns)
    headers = "".join(f"<th>{htmllib.escape(str(col))}</th>" for col in columns)
    rows = []
    for place, (_, row) in enumerate(df.iterrows(), start=1):
        cells = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                rendered = "—"
            elif col == "Equipo":
                rendered = team_name_html(value)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                if col in {"Edad", "Minutos", "Goles", "Asistencias"}:
                    rendered = f"{float(value):,.0f}"
                else:
                    rendered = f"{float(value):,.2f}"
            else:
                rendered = htmllib.escape(str(value))
            cls = "club" if col == "Jugador" else ("team" if col == "Equipo" else "num")
            if col == "Score de perfil":
                cls += " score"
            cells.append(f'<td class="{cls}">{rendered}</td>')
        rows.append(f'<tr style="--row:{place % 2}">{"".join(cells)}</tr>')
    st.markdown(
        "<div class='standings-wrap shortlist-wrap'><table class='standings shortlist-table'><thead><tr>" + headers + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def comparison_table(df: pd.DataFrame) -> None:
    """Tabla de comparación oscura con cabeceras de jugador, club y escudo."""
    headers = []
    for column in df.columns:
        if column == "Métrica":
            headers.append("<th class='compare-metric-head'>Métrica</th>")
            continue
        parts = str(column).rsplit(" · ", 1)
        label = htmllib.escape(parts[0])
        club = team_name_html(parts[1], "compare-team") if len(parts) == 2 else ""
        headers.append(f"<th><div class='compare-player-head'><b>{label}</b>{club}</div></th>")
    rows = []
    for index, (_, row) in enumerate(df.iterrows()):
        cells = []
        for column in df.columns:
            value = row[column]
            if pd.isna(value):
                rendered = "—"
            elif column == "Métrica":
                rendered = htmllib.escape(str(value))
            else:
                number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                rendered = f"{number:,.2f}" if pd.notna(number) else htmllib.escape(str(value))
            cls = "compare-metric" if column == "Métrica" else "compare-value"
            cells.append(f'<td class="{cls}">{rendered}</td>')
        rows.append(f'<tr style="--row:{index % 2}">{"".join(cells)}</tr>')
    st.markdown(
        "<div class='compare-wrap'><table class='compare-table'><thead><tr>" + "".join(headers) +
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def team_comparison_table(df: pd.DataFrame) -> None:
    """Tabla de métricas colectivas con los escudos de los equipos comparados."""
    headers = ["<th class='compare-metric-head'>Métrica</th>"]
    for team in df.columns[1:]:
        headers.append(f"<th>{team_name_html(team, 'compare-team')}</th>")
    rows = []
    for index, (_, row) in enumerate(df.iterrows()):
        cells = [f"<td class='compare-metric'>{htmllib.escape(str(row.iloc[0]))}</td>"]
        for value in row.iloc[1:]:
            number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            rendered = f"{number:,.2f}" if pd.notna(number) else "—"
            cells.append(f"<td class='compare-value'>{rendered}</td>")
        rows.append(f'<tr style="--row:{index % 2}">{"".join(cells)}</tr>')
    st.markdown(
        "<div class='compare-wrap'><table class='compare-table team-compare-table'><thead><tr>" + "".join(headers) +
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def comparison_score_cards(rows: list[pd.Series], scores: list[float | None], colors: dict[str, str]) -> None:
    """Muestra el índice de perfil de cada jugador con el color de su serie."""
    cards = []
    for row, score in zip(rows, scores):
        name = str(row.get("Jugador", ""))
        team = str(row.get("Equipo", ""))
        label = f"{name} · {team}"
        color = htmllib.escape(colors.get(label, LIME))
        role = htmllib.escape(str(row.get("Perfil", "Perfil no disponible")))
        value = f"{score:.1f}" if score is not None else "—"
        club = team_name_html(team, "compare-team")
        cards.append(
            f'<div class="compare-score-card" style="--score-color:{color}">'
            f'<div class="compare-score-label">Índice de perfil · {role}</div>'
            f'<div class="compare-score-value">{value}</div><div class="compare-score-club">{club}</div></div>'
        )
    if cards:
        st.markdown(f'<div class="compare-score-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def player_metrics_table(row: pd.Series) -> None:
    """Presenta las métricas completas del jugador en una tabla compacta y legible."""
    excluded = {"Player", "Nation", "Pos", "Squad", "Age", "Born", "MP", "Starts", "Min", "90s",
                "Jugador", "Nacionalidad", "Posición", "Equipo", "Perfil", "Edad", "Minutos", "Partidos", "Titularidades"}
    categories = {
        "standard": "Estándar", "shooting": "Tiros", "passing": "Pases", "pass_types": "Tipo de pase",
        "creation": "Creación", "goalkeeping": "Portería", "defense": "Defensa", "possession": "Posesión",
        "minutes": "Tiempo de juego", "misc": "Otros",
    }
    labels = {
        "Gls": "Goles", "Ast": "Asistencias", "G+A": "Goles + asistencias", "G-PK": "Goles sin penalti",
        "PK": "Penaltis", "PKatt": "Penaltis lanzados", "xG": "Goles esperados (xG)", "npxG": "xG sin penalti",
        "xAG": "xG asistido", "Sh": "Tiros", "SoT": "Tiros a puerta", "SoT%": "Tiros a puerta (%)",
        "Sh/90": "Tiros/90", "SoT/90": "Tiros a puerta/90", "Cmp": "Pases completados", "Att": "Pases intentados",
        "Cmp%": "Pases completados (%)", "KP": "Pases clave", "PrgP": "Pases progresivos", "PrgC": "Conducciones progresivas",
        "PrgR": "Recepciones progresivas", "Tkl": "Entradas", "TklW": "Entradas ganadas", "Int": "Intercepciones",
        "Clr": "Despejes", "Blocks": "Bloqueos", "Recov": "Recuperaciones", "Won%": "Duelos aéreos ganados (%)",
        "Save%": "Paradas (%)", "CS%": "Porterías a cero (%)", "GA": "Goles encajados", "Min": "Minutos",
        "MP": "Partidos", "Starts": "Titularidades", "90s": "Partidos completos equivalentes", "Age": "Edad",
    }
    pairs = []
    for column, value in row.items():
        if column.startswith("__") or column in excluded or pd.isna(value):
            continue
        if "__" in column:
            category, metric = column.split("__", 1)
            label = f"{categories.get(category, category.title())} · {labels.get(metric, metric)}"
        else:
            label = column
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(number):
            if any(token in label for token in ("/90", "%", "xG", "Índice", "percentil")):
                rendered = f"{number:,.2f}"
            else:
                rendered = f"{number:,.0f}" if float(number).is_integer() else f"{number:,.2f}"
        else:
            rendered = htmllib.escape(str(value))
        pairs.append((label, rendered))
    pairs.sort(key=lambda pair: (pair[0].casefold(), pair[0]))
    body = "".join(
        f'<tr style="--row:{index % 2}"><td>{htmllib.escape(label)}</td><td>{value}</td></tr>'
        for index, (label, value) in enumerate(pairs)
    )
    st.markdown(
        "<div class='player-metrics-wrap'><table class='player-metrics-table'><thead><tr><th>Métrica</th><th>Valor</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def format_hover(fig: go.Figure) -> go.Figure:
    """Presenta tooltips uniformes en español y limita las cifras a dos decimales."""
    for trace in fig.data:
        if trace.type == "bar":
            metric = "x" if trace.orientation == "h" else "y"
            category = "y" if trace.orientation == "h" else "x"
            extra = ""
            customdata = trace.customdata
            if customdata is not None:
                try:
                    custom_array = np.asarray(customdata, dtype=object)
                    if custom_array.ndim > 1:
                        details = []
                        metric_names = list(trace.meta) if isinstance(trace.meta, (list, tuple)) else []
                        for idx, metric_name in enumerate(metric_names):
                            details.append(
                                f"<br><span style='color:#b4c0d3;font-size:11px'>{htmllib.escape(str(metric_name))}</span> "
                                f"<b style='color:#e8eefb'>%{{customdata[{idx}]:,.2f}}</b>"
                            )
                        extra = "".join(details)
                except (TypeError, ValueError):
                    pass
            metric_label = trace.name if trace.name and trace.name != metric else "Valor"
            if metric_label in {"Jugadores", "Pts", "Goles", "GF", "GA", "Partidos", "W", "D", "L", "Rk", "MP"}:
                value_template = f"%{{{metric}:,.0f}}"
            else:
                value_template = f"%{{{metric}:,.2f}}"
            trace.hovertemplate = (
                "<span style='color:#b4c0d3;font-size:11px'>%{fullData.name}</span>"
                f"<br><b style='color:#f4f7fb;font-size:13px'>%{{{category}}}</b>"
                f"<br><span style='color:#b4c0d3;font-size:11px'>{htmllib.escape(metric_label)}</span> "
                f"<b style='color:#b6ff3b;font-size:13px'>{value_template}</b>"
                f"{extra}<extra></extra>"
            )
        elif trace.type == "scatter" and trace.mode and "markers" in trace.mode:
            trace.hovertemplate = (
                "<b style='color:#f4f7fb;font-size:13px'>%{hovertext}</b>"
                "<br><span style='color:#b4c0d3;font-size:11px'>Equipo</span> "
                "<b style='color:#e8eefb'>%{fullData.name}</b>"
                "<br><span style='color:#b4c0d3;font-size:11px'>Goles esperados (xG)</span> "
                "<b style='color:#b6ff3b'>%{x:,.2f}</b>"
                "<br><span style='color:#b4c0d3;font-size:11px'>Goles</span> "
                "<b style='color:#b6ff3b'>%{y:,.2f}</b>"
                "<br><span style='color:#b4c0d3;font-size:11px'>Minutos</span> "
                "<b style='color:#e8eefb'>%{marker.size:,.0f}</b><extra></extra>"
            )
        elif trace.type == "scatterpolar":
            trace.hovertemplate = (
                "<b style='color:#f4f7fb;font-size:13px'>%{fullData.name}</b>"
                "<br>"
                "<span style='color:#b4c0d3;font-size:11px'>Métrica</span> "
                "<b style='color:#f4f7fb;font-size:13px'>%{theta}</b>"
                "<br><span style='color:#b4c0d3;font-size:11px'>Percentil</span> "
                "<b style='color:#b6ff3b;font-size:13px'>%{r:.2f}</b>"
                "<br><span style='color:#b4c0d3;font-size:11px'>Valor</span> "
                "<b style='color:#e8eefb'>%{customdata:.2f}</b><extra></extra>"
            )
    return fig


def _grown(trace, e: float) -> dict:
    """Valores de la traza escalados por e (0→1) para animar su entrada."""
    if trace.type == "bar":
        axis = "x" if trace.orientation == "h" else "y"
        return {axis: (np.asarray(getattr(trace, axis), dtype=float) * e).tolist()}
    if trace.type == "scatterpolar":
        return {"r": (np.asarray(trace.r, dtype=float) * e).tolist()}
    if trace.type == "scatter" and trace.marker.size is not None and not np.isscalar(trace.marker.size):
        return {"marker": {"size": (np.asarray(trace.marker.size, dtype=float) * e).tolist()}}
    return {}


_PLAY_JS = """
var gd = document.getElementById('{plot_id}');
function play(){ Plotly.Plots.resize(gd); Plotly.animate(gd, null, {frame:{duration:28, redraw:true}, transition:{duration:28, easing:'linear'}, mode:'immediate', fromcurrent:false}); }
if ('IntersectionObserver' in window) {
  var io = new IntersectionObserver(function(entries){ if (entries[0].isIntersecting) { io.disconnect(); setTimeout(play, 220); } }, {threshold: 0.2});
  io.observe(gd);
} else { setTimeout(play, 300); }
"""


def animated_chart(fig: go.Figure, height: int, key: str, polar: bool = False) -> None:
    """Aplica el tema oscuro y dibuja el gráfico con animación de entrada (se reproduce al hacerse visible)."""
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT_SECONDARY, size=12),
        margin=dict(l=8, r=8, t=16, b=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#101827", bordercolor="#34445c", font=dict(color="#e8eefb", family="Inter, Segoe UI, sans-serif", size=12), align="left", namelength=-1),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="rgba(255,255,255,.14)", linecolor="rgba(255,255,255,.1)")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="rgba(255,255,255,.14)", linecolor="rgba(255,255,255,.1)")
    if polar:
        fig.update_polars(bgcolor="rgba(255,255,255,.025)",
                          radialaxis=dict(gridcolor="rgba(255,255,255,.13)", linecolor="rgba(255,255,255,.16)", tickfont=dict(size=10, color=TEXT_MUTED)),
                          angularaxis=dict(gridcolor="rgba(255,255,255,.1)", linecolor="rgba(255,255,255,.12)"))
    try:
        fig.update_layout(barcornerradius=7)
    except Exception:
        pass

    f = go.Figure(format_hover(fig))
    # Ejes fijos para que no "salten" mientras se anima desde cero.
    bar_max, bar_h, rad_max, xs, ys = 0.0, True, 0.0, [], []
    for tr in f.data:
        if tr.type == "bar":
            bar_h = tr.orientation == "h"
            vals = np.asarray(tr.x if bar_h else tr.y, dtype=float)
            if vals.size and np.isfinite(vals).any():
                bar_max = max(bar_max, float(np.nanmax(vals)))
        elif tr.type == "scatterpolar":
            vals = np.asarray(tr.r, dtype=float)
            if vals.size and np.isfinite(vals).any():
                rad_max = max(rad_max, float(np.nanmax(vals)))
        elif tr.type == "scatter" and tr.marker.size is not None and not np.isscalar(tr.marker.size):
            xs.append(np.asarray(tr.x, dtype=float))
            ys.append(np.asarray(tr.y, dtype=float))
    if bar_max > 0:
        (f.update_xaxes if bar_h else f.update_yaxes)(range=[0, bar_max * 1.08], autorange=False)
    if rad_max > 0 and f.layout.polar.radialaxis.range is None:
        f.update_polars(radialaxis_range=[0, rad_max * 1.1])
    if xs:
        for arrs, setter in ((xs, f.update_xaxes), (ys, f.update_yaxes)):
            allv = np.concatenate(arrs)
            allv = allv[np.isfinite(allv)]
            if allv.size:
                lo, hi = float(allv.min()), float(allv.max())
                pad = max((hi - lo) * 0.1, 0.5)
                setter(range=[lo - pad, hi + pad], autorange=False)

    idx = [i for i, tr in enumerate(f.data) if _grown(tr, 1.0)]
    if idx:
        # Menos fotogramas ligeros: mantiene una entrada fluida y reduce el HTML enviado.
        steps = 12
        frames = []
        for k in range(1, steps + 1):
            t = k / steps
            e = 1 - (1 - t) ** 3  # ease-out cúbico
            frames.append(go.Frame(data=[{"type": f.data[i].type, **_grown(f.data[i], e)} for i in idx], traces=idx, name=str(k)))
        for i in idx:
            f.data[i].update(_grown(f.data[i], 0.0))
        f.frames = frames
        post_script = _PLAY_JS
    else:
        post_script = "Plotly.Plots.resize(document.getElementById('{plot_id}'));"

    page = f.to_html(full_html=True, include_plotlyjs="cdn", auto_play=False, div_id=f"c-{key}",
                     default_height=f"{height}px", default_width="100%",
                     config={"displayModeBar": False, "responsive": True}, post_script=post_script)
    page = page.replace("<head>", "<head><style>html,body{margin:0;background:transparent;overflow:hidden}</style>", 1)
    if hasattr(st, "iframe"):  # Streamlit reciente (components.v1.html está deprecado)
        st.iframe(page, width="stretch", height=height + 6)
    else:
        components.html(page, height=height + 6, scrolling=False)


def player_score(df: pd.DataFrame, row: pd.Series, min_min: int) -> float | None:
    role = row.get("Perfil")
    if role not in ROLE_METRICS:
        return None
    peers = position_filter(df.copy(), role)
    if "Minutos" in peers:
        peers = peers[numeric(peers["Minutos"]).fillna(0) >= min_min]
    metrics = [m for m in ROLE_METRICS[role] if m in peers.columns and numeric(peers[m]).notna().sum() > 1]
    if peers.empty or not metrics:
        return None
    scores = []
    for metric in metrics:
        values = numeric(peers[metric]).dropna()
        value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
        if len(values) > 1 and pd.notna(value):
            below = int((values < value).sum())
            tied = int((values == value).sum())
            scores.append((below + (tied + 1) / 2) / len(values) * 100)
    return float(np.mean(scores)) if scores else None


def player_card(row: pd.Series, score: float | None) -> None:
    def stat(label: str, key: str, fmt: str) -> str:
        v = row.get(key, pd.NA)
        txt = format(float(v), fmt) if pd.notna(v) else "—"
        return f'<div class="pstat"><b>{txt}</b><span>{label}</span></div>'

    stats = "".join([stat("Edad", "Edad", ".0f"), stat("Minutos", "Minutos", ",.0f"), stat("Goles", "Goles", ".0f"),
                     stat("Asist.", "Asistencias", ".0f"), stat("xG", "xG", ".1f")])
    if score is None:
        ring = '<div class="ring" style="--p:0"><b>—</b><small>Índice</small></div>'
    else:
        ring = f'<div class="ring" style="--p:{score:.0f}"><b class="count" style="--n:{score:.0f}"></b><small>Índice</small></div>'
    tags = "".join(f'<span class="tag">{htmllib.escape(str(t))}</span>' for t in [row.get("Perfil"), row.get("Posición"), row.get("Nacionalidad")]
                   if t is not None and str(t) not in ("", "nan", "<NA>"))
    st.markdown(
        f'<div class="pcard">{ring}<div><div class="pname">{htmllib.escape(str(row["Jugador"]))}</div>'
        f'<div class="pteam">{team_name_html(row["Equipo"], "player-team-name")}</div>{tags}<div class="pstats">{stats}</div></div></div>',
        unsafe_allow_html=True,
    )


# ───────────────────────────── datos y sidebar ─────────────────────────────
existing_leagues = {name: path for name, path in LEAGUE_DIRS.items() if path.is_dir()}
if not existing_leagues:
    st.error("No encuentro las carpetas La-Liga_2024-2025_CSV_Raw ni Ligue-1_2024-2025_CSV_Raw junto a app.py.")
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-logo">⚽</div><div><div class="brand-t">Scout<em>IQ</em></div><div class="brand-s">Player Intelligence</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-h">Contexto</div>', unsafe_allow_html=True)
    league_name = st.selectbox("Competición", list(existing_leagues))
    min_minutes = st.slider("Mínimo de minutos para scouting", 0, 2500, 1800, 90)
    st.caption("Los percentiles se calculan dentro de la competición y perfil seleccionados.")
    section = st.radio("Sección", ["Resumen", "Shortlist", "Ficha de jugador", "Comparar jugadores", "Comparar equipos", "Equipos", "Datos y método"], label_visibility="collapsed")

folder = existing_leagues[league_name]
file_signature = csv_signature(folder)
with st.sidebar:
    if st.button("↻ Recargar CSV ahora", use_container_width=True):
        load_players.clear()
        load_teams.clear()
        st.rerun()

with st.spinner(f"Cargando CSV de {league_name}…"):
    players, player_warnings = load_players(str(folder), file_signature)
    teams, team_warnings = load_teams(str(folder), file_signature)
if csv_signature(folder) != file_signature:
    st.rerun()
if players.empty:
    st.error("No se pudieron combinar los CSV de jugadores. Comprueba que las cabeceras incluyan Player y Squad.")
    st.write(player_warnings)
    st.stop()

players = players.copy()
players["Score de perfil"] = pd.NA
player_count = players["Jugador"].nunique()
team_count = teams["Squad"].nunique() if not teams.empty else 0
minutes_median = numeric(players["Minutos"]).median() if "Minutos" in players else 0
if pd.isna(minutes_median):
    minutes_median = 0

# ───────────────────────────── RESUMEN ─────────────────────────────
if section == "Resumen":
    sec(f"Panorama · {league_name}", "Temporada 2024–2025")
    kpi_cards([("👥", "Jugadores", player_count), ("🏟️", "Equipos", team_count), ("⏱️", "Minutos medianos", minutes_median), ("📅", "Temporada", "2024–25")])
    left, right = st.columns([1.15, 1])
    with left:
        sec("Producción ofensiva", "xG vs goles · tamaño = minutos")
        if "xG" in players and "Goles" in players:
            top = players[numeric(players.get("Minutos", pd.Series(0, index=players.index))) >= min_minutes].copy()
            top["xG"] = numeric(top["xG"])
            top["Goles"] = numeric(top["Goles"])
            top = top.dropna(subset=["xG", "Goles"]).nlargest(18, "xG")
            if not top.empty:
                fig = px.scatter(top, x="xG", y="Goles", color="Equipo", hover_name="Jugador", size="Minutos", size_max=24,
                                 color_discrete_sequence=PALETTE, labels={"xG": "Goles esperados (xG)", "Goles": "Goles", "Minutos": "Minutos"})
                fig.update_layout(showlegend=False)
                fig.add_shape(type="line", x0=0, y0=0, x1=max(1, top["xG"].max()), y1=max(1, top["xG"].max()), line={"dash": "dot", "color": "rgba(255,255,255,.35)"})
                fig.update_traces(marker=dict(opacity=0.9, line=dict(width=1, color="rgba(255,255,255,.6)")))
                animated_chart(fig, 400, "scatter")
            else:
                st.info("No hay suficientes registros sobre el mínimo de minutos elegido.")
        else:
            st.info("No están disponibles xG y goles en estos CSV.")
    with right:
        sec("Distribución por perfil")
        profile_counts = players["Perfil"].value_counts().rename_axis("Perfil").reset_index(name="Jugadores")
        fig = px.bar(profile_counts, x="Jugadores", y="Perfil", orientation="h", color="Jugadores", color_continuous_scale=LIME_SCALE)
        fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        animated_chart(fig, 400, "profiles")
    if not teams.empty and "Pts" in teams:
        sec("Clasificación", "Puntos reales y xG")
        standings_table(teams)

# ───────────────────────────── SHORTLIST ─────────────────────────────
elif section == "Shortlist":
    sec("Shortlist por perfil", "Crea un perfil de contratación con métricas, pesos y umbrales propios")
    scouting_profiles = st.session_state.setdefault("scouting_profiles", {})
    saved_message = st.session_state.pop("scouting_profile_saved_message", None)
    if saved_message:
        st.success(saved_message)
    profile_note = st.session_state.pop("shortlist_profile_note", None)
    if profile_note:
        st.info(profile_note)
    metric_exclusions = {"Jugador", "Equipo", "Nacionalidad", "Posición", "Perfil", "Edad", "Minutos", "Partidos", "Titularidades", "Score de perfil", "Age", "MP", "Starts", "Min", "90s"}
    metric_options = sorted(
        [metric for metric in players.columns
         if not metric.startswith("__") and "__" not in metric and metric not in metric_exclusions
         and numeric(players[metric]).notna().sum() > 1],
        key=str.casefold,
    )
    saved_col, load_col = st.columns([1.5, 1], vertical_alignment="bottom")
    current_league_profiles = [name for name, profile in scouting_profiles.items() if profile.get("league", league_name) == league_name]
    profile_choices = {f"Estándar · {name}": ("standard", name) for name in STANDARD_SCOUT_PROFILES}
    profile_choices.update({f"Guardado · {name}": ("saved", name) for name in current_league_profiles})
    saved_profile_label = saved_col.selectbox("Perfiles estándar y guardados", ["—"] + list(profile_choices), key="shortlist_saved_profile")
    if load_col.button("Cargar perfil", use_container_width=True, disabled=saved_profile_label == "—"):
        profile_kind, profile_name = profile_choices[saved_profile_label]
        if profile_kind == "standard":
            template = STANDARD_SCOUT_PROFILES[profile_name]
            profile_metrics = [metric for metric in template["weights"] if metric in metric_options]
            profile = {
                "league": league_name, "role": template["role"], "age_max": template["age_max"],
                "min_minutes": template["min_minutes"], "metrics": profile_metrics,
                "weights": {metric: template["weights"][metric] for metric in profile_metrics},
                "directions": {metric: metric not in LOWER_IS_BETTER for metric in profile_metrics}, "clubs": [],
            }
            if profile_name == "Delantero presionante":
                st.session_state["shortlist_profile_note"] = "Los CSV no incluyen eventos de presión; este perfil usa recuperaciones, entradas e intercepciones como aproximación defensiva."
        else:
            profile = scouting_profiles[profile_name]
        profile["metrics"] = [metric for metric in profile["metrics"] if metric in metric_options]
        st.session_state["shortlist_role"] = profile["role"]
        st.session_state["shortlist_age_max"] = profile["age_max"]
        st.session_state["shortlist_min_minutes"] = profile["min_minutes"]
        st.session_state["shortlist_mode"] = "Personalizado"
        st.session_state["shortlist_custom_metrics"] = profile["metrics"]
        st.session_state["shortlist_clubs"] = profile.get("clubs", [])
        st.session_state["shortlist_profile_name"] = profile_name
        for metric, weight in profile["weights"].items():
            if metric in profile["metrics"]:
                st.session_state[scouting_metric_key("weight", metric)] = weight
        for metric, higher_is_better in profile["directions"].items():
            if metric in profile["metrics"]:
                st.session_state[scouting_metric_key("direction", metric)] = "Más es mejor" if higher_is_better else "Menos es mejor"
        st.rerun()

    c1, c2, c3 = st.columns([1, 1, 1.4])
    role = c1.selectbox("Perfil de posición", list(ROLE_METRICS), key="shortlist_role")
    age_max = c2.slider("Edad máxima", 16, 40, 28, key="shortlist_age_max")
    clubs = sorted(players["Equipo"].dropna().unique().tolist())
    selected_clubs = c3.multiselect("Equipos", clubs, placeholder="Todos los equipos", key="shortlist_clubs")
    if selected_clubs:
        chips = "".join(team_name_html(club, "team-filter-chip") for club in selected_clubs)
        st.markdown(f"<div class='team-filter-chips'>{chips}</div>", unsafe_allow_html=True)

    profile_mode = st.radio("Criterios del índice", ["Plantilla por posición", "Personalizado"], horizontal=True, key="shortlist_mode")
    profile_min_minutes = st.slider("Mínimo de minutos para este perfil", 0, 2500, min_minutes, 1, key="shortlist_min_minutes")
    metrics = [metric for metric in ROLE_METRICS[role] if metric in metric_options]
    weights = {metric: 1.0 for metric in metrics}
    higher_is_better = {metric: metric not in LOWER_IS_BETTER for metric in metrics}
    if profile_mode == "Personalizado":
        default_custom_metrics = [metric for metric in ROLE_METRICS[role] if metric in metric_options]
        with st.expander("Configurar métricas, pesos y dirección", expanded=True):
            metrics = st.multiselect("Métricas que forman el índice", metric_options, default=default_custom_metrics, key="shortlist_custom_metrics")
            if metrics:
                st.caption("El peso es relativo: un valor mayor da más influencia en el índice. Ajusta la dirección si valores bajos son preferibles.")
                st.markdown("**Criterio por métrica**")
                for metric in metrics:
                    weight_col, metric_col, direction_col = st.columns([1, 2, 1.5])
                    weights[metric] = weight_col.number_input(
                        "Peso", min_value=0.0, max_value=5.0, value=1.0, step=0.25,
                        key=scouting_metric_key("weight", metric),
                    )
                    metric_col.markdown(f"<div class='profile-metric-label'>{htmllib.escape(metric)}</div>", unsafe_allow_html=True)
                    direction = direction_col.selectbox(
                        f"Dirección · {metric}", ["Más es mejor", "Menos es mejor"],
                        index=0 if metric not in LOWER_IS_BETTER else 1,
                        key=scouting_metric_key("direction", metric), label_visibility="collapsed",
                    )
                    higher_is_better[metric] = direction == "Más es mejor"
            st.caption("Los perfiles guardados se conservan durante esta sesión de la app.")
            profile_name = st.text_input("Nombre del perfil", placeholder="Ej. Mediocentro organizador", key="shortlist_profile_name")
            if st.button("Guardar perfil en esta sesión", disabled=not profile_name.strip() or not metrics or not any(float(weights.get(metric, 0.0)) > 0 for metric in metrics)):
                scouting_profiles[profile_name.strip()] = {
                    "league": league_name, "role": role, "age_max": age_max, "min_minutes": profile_min_minutes,
                    "metrics": list(metrics), "weights": {metric: float(weights[metric]) for metric in metrics},
                    "directions": {metric: bool(higher_is_better[metric]) for metric in metrics},
                    "clubs": list(selected_clubs),
                }
                st.session_state["scouting_profiles"] = scouting_profiles
                st.session_state["scouting_profile_saved_message"] = f"Perfil «{profile_name.strip()}» guardado en esta sesión."
                st.rerun()

    pool = position_filter(players.copy(), role)
    if "Minutos" in pool:
        pool = pool[numeric(pool["Minutos"]).fillna(0) >= profile_min_minutes]
    if "Edad" in pool:
        pool = pool[numeric(pool["Edad"]).isna() | (numeric(pool["Edad"]) <= age_max)]
    if selected_clubs:
        pool = pool[pool["Equipo"].isin(selected_clubs)]
    metrics = [metric for metric in metrics if metric in players.columns and numeric(players[metric]).notna().sum() > 1]
    if not metrics:
        st.warning(f"No hay métricas suficientes para construir un índice de {role.lower()} con estos archivos.")
    elif not any(float(weights.get(metric, 0.0)) > 0 for metric in metrics):
        st.warning("Asigna un peso mayor que cero al menos a una métrica para calcular el índice.")
    else:
        peers = position_filter(players.copy(), role)
        if "Minutos" in peers:
            peers = peers[numeric(peers["Minutos"]).fillna(0) >= profile_min_minutes]
        pool = pool.copy()
        if peers.empty:
            st.warning("No hay jugadores de referencia que cumplan el mínimo de minutos para este perfil.")
        pool["Score de perfil"] = weighted_profile_scores(peers, pool, metrics, weights, higher_is_better)
        pool = pool.sort_values("Score de perfil", ascending=False, na_position="last")
        display_cols = [c for c in ["Jugador", "Equipo", "Posición", "Edad", "Minutos", "Goles", "Asistencias", "xG", "npxG", "xAG", *metrics, "Score de perfil"] if c in pool]
        profile_type = "media ponderada de percentiles" if profile_mode == "Personalizado" else "media de percentiles por posición"
        st.caption(f"{len(pool)} jugadores cumplen los filtros · Índice = {profile_type}. Cohorte de referencia: {role.lower()} con al menos {profile_min_minutes:,} minutos.")
        if not pool.empty and pool["Score de perfil"].notna().any():
            best = pool.dropna(subset=["Score de perfil"]).nlargest(15, "Score de perfil").sort_values("Score de perfil", ascending=True)
            fig = px.bar(best, x="Score de perfil", y="Jugador", color="Equipo", orientation="h", color_discrete_sequence=PALETTE,
                         custom_data=[metric for metric in metrics if metric in best], labels={"Score de perfil": "Índice de perfil (0–100)"})
            fig.update_layout(showlegend=False)
            fig.update_traces(marker_color=LIME)
            for trace in fig.data:
                trace.meta = [metric for metric in metrics if metric in best]
            fig.update_layout(bargap=0.28, yaxis={"categoryorder": "array", "categoryarray": best["Jugador"].tolist()})
            animated_chart(fig, 460, "shortlist")
        shortlist_table(pool[display_cols].round(2))
        st.download_button("⬇ Descargar shortlist CSV", pool[display_cols].to_csv(index=False, sep=";", decimal=",", encoding="utf-8-sig"),
                           f"shortlist_{league_name.lower().replace(' ', '-')}_2024-2025.csv", "text/csv")

# ───────────────────────────── FICHA ─────────────────────────────
elif section == "Ficha de jugador":
    sec("Ficha individual", "Percentiles frente a su perfil y a la mediana de la liga")
    choices = player_name_options(players)
    default_player = next((i for i, option in enumerate(choices) if norm(option.rsplit(" · ", 1)[0]) == "lucien agoume"), 0)
    preferred_option = choices[default_player]
    if st.session_state.get("profile_player") == "Max Aarons · Valencia" or st.session_state.get("profile_player") not in choices:
        st.session_state["profile_player"] = preferred_option
    selected = st.selectbox("Jugador", choices, index=default_player, key="profile_player")
    chosen_name, chosen_team = selected.rsplit(" · ", 1)
    row = players[(players["Jugador"] == chosen_name) & (players["Equipo"] == chosen_team)].iloc[0]
    player_card(row, player_score(players, row, min_minutes))

    profile_metrics = ["Goles/90", "npxG/90", "xAG/90", "Tiros a puerta/90", "Pases clave/90", "Pases progresivos/90", "Regates con éxito/90", "Tkl+Int/90", "Recuperaciones/90", "Centros/90"]
    available = [m for m in profile_metrics if m in players and numeric(players[m]).notna().sum() > 1]
    if available:
        role_p = row.get("Perfil")
        peers = position_filter(players.copy(), role_p) if role_p in ROLE_METRICS else players.copy()
        if "Minutos" in peers:
            peers = peers[numeric(peers["Minutos"]).fillna(0) >= min_minutes]
        pct = []
        for m in available:
            v, s = row.get(m), numeric(peers[m]).dropna()
            if pd.notna(v) and len(s) > 1:
                pct.append((m, float((s <= float(v)).mean() * 100)))
        cl, cr = st.columns(2)
        with cl:
            sec("Radar de percentiles", f"vs. {role_p if role_p in ROLE_METRICS else 'toda la liga'}")
            if len(pct) >= 3:
                cats = [p[0] for p in pct]
                vals = [p[1] for p in pct]
                fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], fill="toself", name="Percentil",
                                                line=dict(color=LIME, width=2.5), fillcolor="rgba(182,255,59,.22)", marker=dict(size=7, color=LIME),
                                                customdata=[row.get(m, pd.NA) for m in cats] + [row.get(cats[0], pd.NA)],
                                                hovertemplate=(
                                                    "<b style='color:#f4f7fb'>%{theta}</b>"
                                                    "<br><span style='color:#b4c0d3'>Percentil</span> %{r:.2f}"
                                                    "<br><span style='color:#b4c0d3'>Valor</span> %{customdata:.2f}<extra></extra>"
                                                )))
                fig.update_layout(showlegend=False, polar=dict(radialaxis=dict(range=[0, 100])))
                animated_chart(fig, 430, "ficha-radar", polar=True)
            else:
                st.info("No hay métricas suficientes para el radar.")
        with cr:
            sec("Valor vs mediana de la liga", "Métricas por 90 minutos")
            plot_values = [{"Métrica": m, "Jugador": row.get(m), "Referencia liga": numeric(players[m]).median()} for m in available if pd.notna(row.get(m))]
            if plot_values:
                plot_df = pd.DataFrame(plot_values).melt(id_vars="Métrica", value_vars=["Jugador", "Referencia liga"], var_name="Serie", value_name="Valor")
                plot_df["Serie"] = plot_df["Serie"].replace({"Jugador": "Valor"})
                fig = px.bar(plot_df, x="Valor", y="Métrica", color="Serie", barmode="group", orientation="h",
                             color_discrete_map={"Valor": LIME, "Referencia liga": "#5b6b85"})
                fig.update_layout(legend_title_text="", bargap=0.25)
                animated_chart(fig, 430, "ficha-bars")
    with st.expander("Ver métricas disponibles"):
        player_metrics_table(row)

# ───────────────────────────── COMPARAR ─────────────────────────────
elif section == "Comparar jugadores":
    sec("Comparar jugadores", "Índice de perfil y métricas por 90 minutos")
    names = player_name_options(players)
    preferred_players = {"lucien agoume", "dani ceballos"}
    default_comparison = [option for option in names if norm(option.rsplit(" · ", 1)[0]) in preferred_players]
    if len(default_comparison) != 2:
        default_comparison = names[:2]
    selection = st.multiselect("Elige entre 2 y 5 jugadores", names, default=default_comparison, max_selections=5, key="compare_players")
    compare_metrics = ["Goles/90", "Asistencias/90", "npxG/90", "xAG/90", "Tiros a puerta/90", "Pases clave/90", "Pases progresivos/90", "Regates con éxito/90", "Tkl+Int/90", "Recuperaciones/90", "Duelos aéreos ganados %"]
    chosen_rows = []
    for item in selection:
        name, team = item.rsplit(" · ", 1)
        matches = players[(players["Jugador"] == name) & (players["Equipo"] == team)]
        if not matches.empty:
            chosen_rows.append(matches.iloc[0])
    if len(chosen_rows) < 2:
        st.info("Selecciona al menos dos jugadores para compararlos.")
    else:
        comparison = pd.DataFrame(chosen_rows)
        labels = [f"{r['Jugador']} · {r['Equipo']}" for _, r in comparison.iterrows()]
        comparison_colors = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)}
        scores = [player_score(players, row, min_minutes) for row in chosen_rows]
        comparison_score_cards(chosen_rows, scores, comparison_colors)
        show_metrics = [m for m in compare_metrics if m in comparison and numeric(comparison[m]).notna().any()]
        if show_metrics:
            base = players[numeric(players["Minutos"]).fillna(0) >= min_minutes] if "Minutos" in players else players
            long_rows = []
            for lab, r in zip(labels, chosen_rows):
                for m in show_metrics:
                    v = pd.to_numeric(r.get(m), errors="coerce")
                    s = numeric(base[m]).dropna() if m in base else pd.Series(dtype=float)
                    long_rows.append({"Jugador · equipo": lab, "Métrica": m, "Valor": v,
                                      "Percentil": float((s <= v).mean() * 100) if pd.notna(v) and len(s) > 1 else 0.0})
            long = pd.DataFrame(long_rows)
            fig = px.line_polar(long, r="Percentil", theta="Métrica", color="Jugador · equipo", line_close=True, color_discrete_map=comparison_colors, custom_data=["Valor"])
            fig.update_traces(
                fill="toself", opacity=0.72, line=dict(width=2.5),
                hovertemplate=(
                    "<b style='color:#f4f7fb;font-size:13px'>%{fullData.name}</b>"
                    "<br><span style='color:#b4c0d3;font-size:11px'>Métrica</span> %{theta}"
                    "<br><span style='color:#b4c0d3;font-size:11px'>Percentil</span> "
                    "<b style='color:#b6ff3b'>%{r:.2f}</b>"
                    "<br><span style='color:#b4c0d3;font-size:11px'>Valor</span> "
                    "<b style='color:#e8eefb'>%{customdata:.2f}</b><extra></extra>"
                ),
            )
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), showlegend=False)
            animated_chart(fig, 540, "compare", polar=True)
            player_crest_badges(chosen_rows, comparison_colors)
        table_df = pd.DataFrame({"Métrica": show_metrics, **{labels[i]: [numeric(comparison.iloc[i:i + 1][m]).iloc[0] for m in show_metrics] for i in range(len(labels))}})
        comparison_table(table_df)
        st.caption("Las métricas por 90 minutos ayudan a ajustar diferencias de tiempo jugado. Los roles y el contexto táctico siguen siendo importantes.")

# ───────────────────────────── EQUIPOS ─────────────────────────────
elif section == "Comparar equipos":
    sec("Comparar equipos", "Perfil colectivo en percentiles y valores absolutos")
    if teams.empty:
        st.warning("No hay tablas de equipos legibles para esta competición.")
    else:
        team_options = teams["Squad"].dropna().astype(str).tolist()
        preferred_teams = {"La Liga": ["Barcelona", "Real Madrid"], "Ligue 1": ["Paris S-G", "Marseille"]}
        default_teams = [name for name in preferred_teams.get(league_name, []) if name in team_options]
        if len(default_teams) < 2:
            default_teams = team_options[:2]
        selected_teams = st.multiselect("Equipos a comparar", team_options, default=default_teams, max_selections=5)
        if len(selected_teams) < 2:
            st.info("Selecciona al menos dos equipos para ver la comparación.")
        else:
            available_metrics = [metric for metric in TEAM_COMPARE_METRICS if metric in teams and numeric(teams[metric]).notna().sum() > 1]
            if not available_metrics:
                st.warning("No hay métricas colectivas suficientes para comparar estos equipos.")
            else:
                default_metrics = [metric for metric in ["xG equipo", "xGA equipo", "Posesión %", "Tiros a puerta %", "Pases progresivos/90", "SCA/90"] if metric in available_metrics]
                selected_metrics = st.multiselect("Métricas del perfil", available_metrics, default=default_metrics, max_selections=9)
                team_colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(selected_teams)}
                if len(selected_metrics) < 3:
                    st.info("Selecciona al menos tres métricas para construir el perfil colectivo.")
                else:
                    fig = go.Figure()
                    for team in selected_teams:
                        team_rows = teams[teams["Squad"] == team]
                        if team_rows.empty:
                            continue
                        row = team_rows.iloc[0]
                        percentiles, raw_values = [], []
                        for metric in selected_metrics:
                            values = numeric(teams[metric])
                            ranking = values.rank(pct=True, ascending=(metric not in {"xGA equipo", "GA"})) * 100
                            team_index = team_rows.index[0]
                            ranked_value = ranking.loc[team_index]
                            percentiles.append(float(ranked_value) if pd.notna(ranked_value) else 0.0)
                            raw_values.append(float(row[metric]) if pd.notna(row[metric]) else np.nan)
                        color = team_colors[team]
                        red, green, blue = (int(color[pos:pos + 2], 16) for pos in (1, 3, 5))
                        fig.add_trace(go.Scatterpolar(
                            r=percentiles + percentiles[:1], theta=selected_metrics + selected_metrics[:1],
                            customdata=raw_values + raw_values[:1], name=team, fill="toself", opacity=.68,
                            line=dict(color=color, width=2.4), fillcolor=f"rgba({red},{green},{blue},.13)",
                            hovertemplate=(
                                "<b style='color:#f4f7fb'>%{fullData.name}</b>"
                                "<br><span style='color:#b4c0d3'>Métrica</span> %{theta}"
                                "<br><span style='color:#b4c0d3'>Percentil</span> <b style='color:#b6ff3b'>%{r:.2f}</b>"
                                "<br><span style='color:#b4c0d3'>Valor</span> <b style='color:#e8eefb'>%{customdata:.2f}</b><extra></extra>"
                            ),
                        ))
                    fig.update_layout(showlegend=False, polar=dict(radialaxis=dict(range=[0, 100])))
                    animated_chart(fig, 520, "compare-teams", polar=True)
                    team_crest_badges(selected_teams, team_colors)
                    metric_rows = [
                        {"Métrica": metric, **{team: teams.loc[teams["Squad"] == team, metric].iloc[0] for team in selected_teams}}
                        for metric in selected_metrics
                    ]
                    team_comparison_table(pd.DataFrame(metric_rows))
                    st.caption("xGA y goles concedidos se invierten en el radar: un percentil alto indica menos ocasiones o goles concedidos. La tabla muestra los valores originales.")

elif section == "Equipos":
    sec("Análisis colectivo", "Estilo de juego y ranking por métrica")
    if teams.empty:
        st.warning("No hay tablas de equipos legibles para esta competición.")
    else:
        team_list = teams["Squad"].dropna().astype(str).tolist()
        selected_team = st.selectbox("Equipo", team_list)
        st.markdown(f"<div class='team-select-card'>{team_name_html(selected_team)}</div>", unsafe_allow_html=True)
        team_row = teams[teams["Squad"] == selected_team].iloc[0]
        left, right = st.columns([1, 1.2])
        with left:
            cols = st.columns(3)
            for i, (label, key) in enumerate([("Puntos", "Pts"), ("Goles", "GF"), ("xG", "xG equipo"), ("Posesión", "Posesión %"), ("Tiros/90", "Tiros/90"), ("Pases clave", "Pases clave/90")]):
                value = team_row.get(key, pd.NA)
                cols[i % 3].metric(label, f"{float(value):.1f}" if pd.notna(value) else "—")
            style_metrics = ["xG equipo", "Tiros a puerta %", "Posesión %", "Pases progresivos/90", "Pases clave/90", "SCA/90", "Tkl+Int"]
            avail = [m for m in style_metrics if f"pct__{m}" in teams.columns and pd.notna(team_row.get(f"pct__{m}"))]
            if avail:
                rr = [float(team_row[f"pct__{m}"]) for m in avail]
                fig = go.Figure(go.Scatterpolar(r=rr + rr[:1], theta=avail + avail[:1], fill="toself", line=dict(color=LIME, width=2.5),
                                                fillcolor="rgba(182,255,59,.22)", marker=dict(size=7, color=LIME),
                                                customdata=[team_row.get(m) for m in avail] + [team_row.get(avail[0])],
                                                hovertemplate=(
                                                    "<b style='color:#f4f7fb'>%{theta}</b>"
                                                    "<br><span style='color:#b4c0d3'>Percentil</span> %{r:.2f}"
                                                    "<br><span style='color:#b4c0d3'>Valor</span> %{customdata:.2f}<extra></extra>"
                                                )))
                fig.update_layout(showlegend=False, polar=dict(radialaxis=dict(range=[0, 100])))
                animated_chart(fig, 410, "team-radar", polar=True)
        with right:
            sec("Perfil de la competición")
            metric_choice = st.selectbox("Ordenar equipos por", [c for c in ["Pts", "xG equipo", "xGA equipo", "Posesión %", "Tiros/90", "Tiros a puerta %", "Pases progresivos/90", "Pases clave/90", "SCA/90", "Tkl+Int", "Recuperaciones"] if c in teams.columns])
            ranked = teams.sort_values(metric_choice, ascending=metric_choice == "xGA equipo").head(20).copy()
            fig = px.bar(ranked, x=metric_choice, y="Squad", orientation="h", color=metric_choice, color_continuous_scale=LIME_SCALE)
            fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"}, bargap=0.22)
            add_team_crests_to_bar_axis(fig, ranked["Squad"].astype(str).tolist())
            animated_chart(fig, 540, "team-bars")
        sec("Tabla de equipos", "Concesión de ocasiones incluida")
        standings_table(teams)

# ───────────────────────────── DATOS Y MÉTODO ─────────────────────────────
elif section == "Datos y método":
    sec("Datos, cobertura y método")
    st.write(f"**Origen:** CSV locales de `{folder.name}`. **Temporada:** 2024–2025. La app no realiza peticiones a FBref ni actualiza estos datos.")
    st.write("**Ficheros de jugadores detectados**")
    file_rows = []
    for category, prefix in {"Estándar": "players_standard", **{k.replace("_", " ").title(): v for k, v in PLAYER_SOURCES.items()}}.items():
        path = latest_csv(folder, prefix)
        file_rows.append({"Grupo": category, "Archivo": path.name if path else "No disponible", "Fecha del archivo": path.stem.rsplit("_", 1)[-1] if path else "—"})
    table(pd.DataFrame(file_rows))
    st.markdown("**Cómo leer los indicadores**")
    st.markdown("""
    - **Por 90**: total de la temporada dividido por los 90 minutos disputados (`90s`). Reduce el sesgo por minutos, pero no elimina diferencias de rol ni de equipo.
    - **Índice de perfil**: media de percentiles de las métricas disponibles para el perfil seleccionado. Se compara con jugadores del mismo perfil y liga, tras aplicar el mínimo de minutos. Es una herramienta de priorización, no una nota de calidad ni una probabilidad de adaptación.
    - **xG y xAG**: estimaciones de calidad de ocasiones y de creación de ocasiones para remates, respectivamente. `npxG` excluye penaltis.
    - **Contexto**: la ficha resume una temporada; no contiene vídeo, datos físicos, tracking, historial médico, coste contractual, pie dominante ni observación cualitativa.
    - **Comparación entre ligas**: los percentiles de shortlist se calculan por separado dentro de cada liga. Los valores absolutos entre La Liga y Ligue 1 no corrigen diferencias de estilo o nivel.
    """)
    all_warnings = player_warnings + team_warnings
    if all_warnings:
        st.warning("Hay diferencias entre los archivos disponibles; algunas métricas pueden quedar vacías.")
        with st.expander("Ver avisos de cobertura"):
            st.write("\n".join(f"- {item}" for item in all_warnings))

st.markdown('<div class="foot">Hecho con <b>Streamlit</b> · <b>Plotly</b> · Datos FBref 2024–25</div>', unsafe_allow_html=True)
