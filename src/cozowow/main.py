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
    Build a stored relation access atom while preserving the order of columns.
    For each column in the select list, if a where condition exists for that column, output "col: <value>"
    Otherwise, output the column name.
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
    ) -> pd.DataFrame:
        relation_access = build_stored_relation_access(from_, select, where, validity)
        head = "?[" + ", ".join(select) + "] :="
        atoms = [relation_access]
        if conditions:
            atoms.extend(conditions)
        rule_body = head + " " + ", ".join(atoms)

        options = []
        if order_by:
            if isinstance(order_by, list):
                options.append(":order " + ", ".join(order_by))
            else:
                options.append(f":order {order_by}")
        if offset is not None:
            options.append(f":offset {offset}")
        if limit is not None:
            options.append(f":limit {limit}")

        script = rule_body + ("\n" + "\n".join(options) if options else "\n")
        return self.script(script)

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
        raw_result = db.script("?[] <- [['hello', 'world', 'Cozo!']]")
        print("Raw Query Result:")
        print(raw_result)

        # Non-temporal relation.
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
        )
        print("\nNon-Temporal Query Result:")
        print(result)

        # Temporal relation.
        db.create_relation("hos", keys=["state", "year"], values=["hos"], temporal=True)
        hos_data = [
            {"state": "US", "year": [2001, True], "hos": "Bush"},
            {"state": "US", "year": [2009, True], "hos": "Obama"},
            {"state": "US", "year": [2017, True], "hos": "Trump"},
            {"state": "US", "year": [2021, True], "hos": "Biden"},
        ]
        db.put("hos", hos_data)
        # According to the tutorial, a correct temporal query is:
        # ?[hos, year] := *hos{state: "US", year, hos @ 2019}
        temporal_result = db.query(
            select=["hos", "year"], from_="hos", where={"state": "US"}, validity=2019
        )
        print("\nTemporal Query Result (2019):")
        print(temporal_result)

        # Current temporal query using NOW.
        current_result = db.query(
            select=["hos", "year"], from_="hos", where={"state": "US"}, validity="NOW"
        )
        print("\nCurrent Temporal Query Result:")
        print(current_result)
