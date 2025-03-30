import pandas as pd
from typing import Any, Dict, List, Union, Optional
from loguru import logger
from pycozo.client import Client


def format_value(val: Any) -> str:
    if isinstance(val, str):
        return f'"{val.replace("\"", "\\\"")}"'
    elif isinstance(val, bool):
        return "true" if val else "false"
    elif val is None:
        return "null"
    else:
        return str(val)


def format_validity(val: Any) -> str:
    if isinstance(val, str) and val.upper() in {"NOW", "END", "ASSERT", "RETRACT"}:
        return f"'{val.upper()}'"
    elif isinstance(val, (int, list)):
        return str(val)
    return format_value(val)


def build_stored_relation_access(
    name: str,
    columns: List[str],
    where: Optional[Dict[str, Any]] = None,
    validity: Optional[Any] = None,
) -> str:
    """
    Build a stored relation access atom.

    For each column in the select list, if a where condition exists for that column,
    output "col: <value>"; otherwise, output the column name.
    If a validity clause is provided, append it (using '@') to the last column.
    """
    col_reprs = []
    for col in columns:
        if where and col in where:
            col_reprs.append(f"{col}: {format_value(where[col])}")
        else:
            col_reprs.append(col)
    if validity is not None and col_reprs:
        validity_str = format_validity(validity)
        col_reprs[-1] = f"{columns[-1]} @ {validity_str}"
    cols_str = ", ".join(col_reprs)
    return f"*{name}{{{cols_str}}}"


def build_inline_rule(
    head_vars: List[str],
    atoms: List[str],
    query_options: Optional[Dict[str, Any]] = None,
    rule_name: str = "?",
) -> str:
    """
    Build an inline CozoScript rule.

    The head is constructed as:
        rule_name[<columns>] :=
    and the body is the conjunction of atoms joined with commas.

    Query options (such as :limit, :offset, :timeout, :sleep, :order/:sort, :assert)
    are appended as separate lines.
    """
    head = f"{rule_name}[{', '.join(head_vars)}] :="
    body = ", ".join(atoms)
    rule = head + " " + body
    if query_options:
        # Each option is added on its own new line
        options_str = "\n".join(f":{k} {v}" for k, v in query_options.items())
        rule += "\n" + options_str
    return rule


def build_fixed_rule(
    head_vars: List[str],
    utility: str,
    input_relations: List[str],
    options: Optional[Dict[str, Any]] = None,
    rule_name: str = "?",
) -> str:
    """
    Build a fixed CozoScript rule.

    A fixed rule uses the syntax:
        rule_name[<columns>] <~ Utility(*relation1, *relation2, ..., option1: value, ...)

    The arity of the output is determined by the utility.
    """
    head = f"{rule_name}[{', '.join(head_vars)}] <~ "
    # Input relations are assumed to be stored relations already formatted (with a leading asterisk)
    inputs = ", ".join(input_relations)
    rule = head + f"{utility}({inputs}"
    if options:
        opts = ", ".join(f"{k}: {format_value(v)}" for k, v in options.items())
        rule += f", {opts}"
    rule += ")"
    return rule


class CozoDB:
    def __init__(
        self,
        engine: str = "sqlite",
        db_path: str = "mydb.db",
        dataframe: bool = True,
        **options,
    ):
        self.client = Client(engine, db_path, dataframe=dataframe, **options)

    def script(
        self, script: str, params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        if params is None:
            params = {}
        logger.debug(f"Executing query:\n{script}")
        return self.client.run(script, params)

    def create_relation(
        self,
        name: str,
        keys: List[str],
        values: List[str] = None,
        temporal: bool = False,
    ) -> None:
        keys_str = ", ".join(
            f"{k}: Validity" if (temporal and k == keys[-1]) else k for k in keys
        )
        schema = keys_str
        if values:
            values_str = ", ".join(values)
            schema = f"{keys_str} => {values_str}"
        query = f":create {name} {{{schema}}}"
        self.script(query)

    def query(
        self,
        select: List[str],
        from_: str,
        where: Optional[Dict[str, Any]] = None,
        conditions: Optional[List[str]] = None,
        order_by: Optional[Union[str, List[str]]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        validity: Optional[Any] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Build and execute an inline CozoScript query.

        This method now supports additional query options as defined in CozoScript:
        :limit, :offset, :timeout, :sleep, :order/:sort, and :assert.
        """
        relation_access = build_stored_relation_access(from_, select, where, validity)
        head = "?[" + ", ".join(select) + "] :="
        atoms = [relation_access]
        if conditions:
            atoms.extend(conditions)
        # Build query options dictionary from standard params and any extras.
        options: Dict[str, Any] = {}
        if order_by:
            if isinstance(order_by, list):
                options["order"] = ", ".join(order_by)
            else:
                options["order"] = order_by
        if offset is not None:
            options["offset"] = offset
        if limit is not None:
            options["limit"] = limit
        if extra_options:
            options.update(extra_options)
        rule = build_inline_rule(head_vars=select, atoms=atoms, query_options=options)
        return self.script(rule)

    def fixed_query(
        self,
        head_vars: List[str],
        utility: str,
        input_relations: List[str],
        options: Optional[Dict[str, Any]] = None,
        rule_name: str = "?",
    ) -> pd.DataFrame:
        """
        Build and execute a fixed CozoScript query.

        Fixed rules are used to call utilities or algorithms such as PageRank,
        DFS, etc. The input_relations list should already include any necessary
        formatting (e.g. an asterisk for stored relations).
        """
        rule = build_fixed_rule(head_vars, utility, input_relations, options, rule_name)
        return self.script(rule)

    def put(
        self,
        relation: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        validity_field: Optional[str] = None,
        validity_value: Optional[Any] = None,
    ) -> pd.DataFrame:
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")
        if validity_field and validity_value is not None:
            for row in data:
                row.setdefault(validity_field, validity_value)
        return self.client.put(relation, data)

    def remove(
        self, relation: str, keys: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> pd.DataFrame:
        if isinstance(keys, dict):
            keys = [keys]
        return self.client.rm(relation, keys)

    def drop_relation(self, name: str) -> pd.DataFrame:
        return self.script(f"::remove {name}")

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "CozoDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


if __name__ == "__main__":
    with CozoDB(engine="sqlite", db_path="mydb.db") as db:
        # Example: simple constant rule (syntax sugar for a fixed rule)
        raw_result = db.script("?[] <- [['hello', 'world', 'Cozo!']]")
        print("Raw Query Result:")
        print(raw_result)

        # Non-temporal relation: airport.
        db.create_relation(
            "airport", keys=["code"], values=["desc", "lon", "lat", "country"]
        )
        sample_data = [
            {
                "code": "JFK",
                "desc": "John F. Kennedy",
                "lon": -73.7781,
                "lat": 40.6413,
                "country": "US",
            },
            {
                "code": "LAX",
                "desc": "Los Angeles",
                "lon": -118.4085,
                "lat": 33.9416,
                "country": "US",
            },
        ]
        db.put("airport", sample_data)
        result = db.query(
            select=["code", "desc", "lon", "lat"],
            from_="airport",
            where={"country": "US"},
            conditions=["lon > -0.1", "lon < 0.1"],
            order_by="code",
            limit=5,
            extra_options={"timeout": 60, "assert": "none"},
        )
        print("\nNon-Temporal Query Result:")
        print(result)

        # Temporal relation: hos.
        db.create_relation("hos", keys=["state", "year"], values=["hos"], temporal=True)
        hos_data = [
            {"state": "US", "year": [2001, True], "hos": "Bush"},
            {"state": "US", "year": [2009, True], "hos": "Obama"},
            {"state": "US", "year": [2017, True], "hos": "Trump"},
            {"state": "US", "year": [2021, True], "hos": "Biden"},
        ]
        db.put("hos", hos_data)
        temporal_result = db.query(
            select=["hos", "year"], from_="hos", where={"state": "US"}, validity=2019
        )
        print("\nTemporal Query Result (2019):")
        print(temporal_result)

        current_result = db.query(
            select=["hos", "year"], from_="hos", where={"state": "US"}, validity="NOW"
        )
        print("\nCurrent Temporal Query Result:")
        print(current_result)

        # Create and populate the 'route' relation for the fixed query.
        db.create_relation("route", keys=["src", "dst"], values=["weight"])
        sample_routes = [
            {"src": "A", "dst": "B", "weight": 1.0},
            {"src": "B", "dst": "C", "weight": 2.0},
            {"src": "C", "dst": "A", "weight": 3.0},
        ]
        db.put("route", sample_routes)

        # Fixed query using a utility (e.g., PageRank).
        fixed_result = db.fixed_query(
            head_vars=[],  # Adjust head variables according to the utility's output arity.
            utility="PageRank",
            input_relations=["*route[]"],
            options={"theta": 0.5},
        )
        print("\nFixed Query (PageRank) Result:")
        print(fixed_result)
