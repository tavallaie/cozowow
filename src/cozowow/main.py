import pandas as pd
from typing import Any, Dict, List, Union, Optional
from dataclasses import dataclass, field
from loguru import logger
from pycozo.client import Client


def format_value(val: Any) -> str:
    """Format a Python value as a CozoScript literal, with special handling for Validity tuples."""
    if isinstance(val, str):
        return f'"{val.replace("\"", "\\\"")}"'
    elif isinstance(val, bool):
        return "true" if val else "false"
    elif val is None:
        return "null"
    elif (
        isinstance(val, list)
        and len(val) == 2
        and isinstance(val[0], (int, float))
        and isinstance(val[1], bool)
    ):
        return f"[{val[0]}, {'true' if val[1] else 'false'}]"
    else:
        return str(val)


def format_validity(val: Any) -> str:
    """Format validity keywords or values for temporal queries."""
    if isinstance(val, str) and val.upper() in {"NOW", "END", "ASSERT", "RETRACT"}:
        return f"'{val.upper()}'"
    elif isinstance(val, (int, list)):
        return str(val)
    return format_value(val)


@dataclass
class QueryOptions:
    """Options for configuring a query's behavior."""

    order: Optional[Union[str, List[str]]] = None
    offset: Optional[int] = None
    limit: Optional[int] = None
    timeout: Optional[int] = None
    sleep: Optional[int] = None
    assert_: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        opts: Dict[str, Any] = {}
        if self.order:
            if isinstance(self.order, list):
                opts["order"] = ", ".join(self.order)
            else:
                opts["order"] = self.order
        if self.offset is not None:
            opts["offset"] = self.offset
        if self.limit is not None:
            opts["limit"] = self.limit
        if self.timeout is not None:
            opts["timeout"] = self.timeout
        if self.sleep is not None:
            opts["sleep"] = self.sleep
        if self.assert_ is not None:
            opts["assert"] = self.assert_
        opts.update(self.extra)
        return opts


@dataclass
class RelationSpec:
    """Specification for a stored relation's schema."""

    name: str
    keys: List[str]
    values: List[str] = field(default_factory=list)
    temporal: bool = False

    def schema(self) -> str:
        keys_str = ", ".join(
            f"{k}: Validity" if self.temporal and k == self.keys[-1] else k
            for k in self.keys
        )
        if self.values:
            values_str = ", ".join(self.values)
            return f"{keys_str} => {values_str}"
        return keys_str


@dataclass
class ChainQuery:
    """Represents a single query in a chained transaction."""

    query: str
    options: Optional[QueryOptions] = None

    def to_script(self) -> str:
        opts = self.options.as_dict() if self.options else {}
        script = "{\n" + self.query
        if opts:
            script += "\n" + "\n".join(f":{k} {v}" for k, v in opts.items())
        script += "\n}"
        return script


class CozoDB:
    """A Pythonic wrapper for CozoDB with advanced features."""

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
        """Execute a raw CozoScript query."""
        if params is None:
            params = {}
        logger.debug(f"Executing query:\n{script}")
        try:
            return self.client.run(script, params)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def create_relation(self, spec: RelationSpec, query: Optional[str] = None) -> None:
        """Create a stored relation with an optional initial query."""
        op = f":create {spec.name} {{{spec.schema()}}}"
        if query:
            op += "\n" + query
        self.script(op)

    def replace_relation(self, spec: RelationSpec, query: str) -> None:
        """Replace a stored relation with new data from a query."""
        op = f":replace {spec.name} {{{spec.schema()}}}\n" + query
        self.script(op)

    @staticmethod
    def build_mutation_spec(spec: RelationSpec) -> str:
        """Build a mutation spec string from a RelationSpec."""
        keys_str = ", ".join(spec.keys)
        if spec.values:
            values_str = ", ".join(spec.values)
            return f"{{{keys_str} => {values_str}}}"
        return f"{{{keys_str}}}"

    def mutate_relation(
        self,
        op_name: str,
        relation: str,
        spec: RelationSpec,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        returning: bool = False,
    ) -> pd.DataFrame:
        """Perform a mutation operation with schema-aware spec and multi-row support."""
        if isinstance(data, dict):
            data = [data]
        if not data:
            raise ValueError("Data cannot be empty")

        columns = spec.keys + spec.values
        required_keys = set(spec.keys)

        for row in data:
            if not isinstance(row, dict):
                raise ValueError(f"Each data item must be a dict, got {type(row)}")
            missing_keys = required_keys - set(row.keys())
            if missing_keys:
                raise ValueError(f"Missing required keys {missing_keys} in row {row}")

        rows = []
        for row in data:
            row_values = [format_value(row.get(col, None)) for col in columns]
            rows.append(f"[{', '.join(row_values)}]")
        constant_rule = f"?[{', '.join(columns)}] <- [{', '.join(rows)}]"
        spec_str = self.build_mutation_spec(spec)
        script = f"{constant_rule}\n:{op_name} {relation} {spec_str}"
        if returning:
            script += " :returning"
        return self.script(script)

    def put_relation(
        self,
        relation: str,
        spec: RelationSpec,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        returning: bool = False,
    ) -> pd.DataFrame:
        """Put rows into a relation with schema awareness and multi-row support."""
        return self.mutate_relation("put", relation, spec, data, returning)

    def remove_rows(
        self,
        relation: str,
        spec: RelationSpec,
        keys: Union[Dict[str, Any], List[Dict[str, Any]]],
        returning: bool = False,
    ) -> pd.DataFrame:
        """Remove rows from a relation using keys."""
        if isinstance(keys, dict):
            keys = [keys]
        if not keys:
            raise ValueError("Keys cannot be empty")

        key_columns = spec.keys
        required_keys = set(key_columns)

        for key_dict in keys:
            if not isinstance(key_dict, dict):
                raise ValueError(f"Each key item must be a dict, got {type(key_dict)}")
            missing_keys = required_keys - set(key_dict.keys())
            if missing_keys:
                raise ValueError(
                    f"Missing required keys {missing_keys} in key {key_dict}"
                )

        rows = [
            [format_value(key_dict.get(col, None)) for col in key_columns]
            for key_dict in keys
        ]
        constant_rule = f"?[{', '.join(key_columns)}] <- [{', '.join(f'[{', '.join(row)}]' for row in rows)}]"
        spec_str = "{" + ", ".join(key_columns) + "}"
        script = f"{constant_rule}\n:rm {relation} {spec_str}"
        if returning:
            script += " :returning"
        return self.script(script)

    def insert_relation(
        self,
        relation: str,
        spec: RelationSpec,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        returning: bool = False,
    ) -> pd.DataFrame:
        """Insert rows into a relation (fails if keys exist)."""
        return self.mutate_relation("insert", relation, spec, data, returning)

    def update_relation(
        self,
        relation: str,
        spec: RelationSpec,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        returning: bool = False,
    ) -> pd.DataFrame:
        """Update rows in a relation (only specified non-keys updated)."""
        return self.mutate_relation("update", relation, spec, data, returning)

    def delete_relation(
        self,
        relation: str,
        spec: RelationSpec,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        returning: bool = False,
    ) -> pd.DataFrame:
        """Delete rows from a relation (only specified non-keys deleted)."""
        return self.mutate_relation("delete", relation, spec, data, returning)

    def ensure_relation(
        self,
        relation: str,
        spec: RelationSpec,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        returning: bool = False,
    ) -> pd.DataFrame:
        """Ensure rows exist in the relation for transaction consistency."""
        return self.mutate_relation("ensure", relation, spec, data, returning)

    def ensure_not_relation(
        self,
        relation: str,
        spec: RelationSpec,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """Ensure rows do not exist in the relation for transaction consistency."""
        return self.mutate_relation("ensure_not", relation, spec, data)

    @staticmethod
    def build_stored_relation_access(
        name: str,
        columns: List[str],
        where: Optional[Dict[str, Any]] = None,
        validity: Optional[Any] = None,
    ) -> str:
        """Build a stored relation access atom."""
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

    @staticmethod
    def build_inline_rule(
        head_vars: List[str],
        atoms: List[str],
        query_options: Optional[Dict[str, Any]] = None,
        rule_name: str = "?",
    ) -> str:
        """Build an inline Datalog rule."""
        head = f"{rule_name}[{', '.join(head_vars)}] :="
        body = ", ".join(atoms)
        rule = head + " " + body
        if query_options:
            opts_str = "\n".join(f":{k} {v}" for k, v in query_options.items())
            rule += "\n" + opts_str
        return rule

    def query(
        self,
        select: List[str],
        from_: str,
        where: Optional[Dict[str, Any]] = None,
        conditions: Optional[List[str]] = None,
        options: Optional[QueryOptions] = None,
        validity: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Execute a query with optional temporal support."""
        relation_access = self.build_stored_relation_access(
            from_, select, where, validity
        )
        atoms = [relation_access]
        if conditions:
            atoms.extend(conditions)
        opts = options.as_dict() if options else {}
        rule = self.build_inline_rule(head_vars=select, atoms=atoms, query_options=opts)
        return self.script(rule)

    def chain_queries(self, queries: List[ChainQuery]) -> pd.DataFrame:
        """Execute a chain of queries in a single transaction."""
        script = "\n".join(q.to_script() for q in queries)
        return self.script(script)

    def create_index(self, relation: str, index_name: str, columns: List[str]) -> None:
        """Create an index on a stored relation."""
        cols_str = ", ".join(columns)
        script = f"::index create {relation}:{index_name} {{{cols_str}}}"
        self.script(script)

    def drop_index(self, relation: str, index_name: str) -> None:
        """Drop an index from a stored relation."""
        script = f"::index drop {relation}:{index_name}"
        self.script(script)

    def set_triggers(
        self,
        relation: str,
        on_put: Optional[List[str]] = None,
        on_rm: Optional[List[str]] = None,
        on_replace: Optional[List[str]] = None,
    ) -> None:
        """Set triggers on a stored relation."""
        triggers = []
        if on_put:
            for q in on_put:
                triggers.append(f"on put {{ {q} }}")
        if on_rm:
            for q in on_rm:
                triggers.append(f"on rm {{ {q} }}")
        if on_replace:
            for q in on_replace:
                triggers.append(f"on replace {{ {q} }}")
        if not triggers:
            script = f"::set_triggers {relation}"
        else:
            script = f"::set_triggers {relation}\n" + "\n".join(triggers)
        self.script(script)

    def put(
        self,
        relation: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        validity_field: Optional[str] = None,
        validity_value: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Insert or update data using the client API (no returning option)."""
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
        """Remove data using the client API (no returning option)."""
        if isinstance(keys, dict):
            keys = [keys]
        return self.client.rm(relation, keys)

    def drop_relation(self, name: str) -> pd.DataFrame:
        """Drop a stored relation entirely."""
        return self.script(f"::remove {name}")

    def close(self) -> None:
        """Close the database connection."""
        self.client.close()

    def __enter__(self) -> "CozoDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


if __name__ == "__main__":
    # Define relation specs
    airport_spec = RelationSpec(
        name="airport", keys=["code"], values=["desc", "lon", "lat", "country"]
    )
    hos_spec = RelationSpec(
        name="hos", keys=["state", "year"], values=["hos"], temporal=True
    )
    route_spec = RelationSpec(name="route", keys=["src", "dst"], values=["weight"])
    rel_spec = RelationSpec(name="rel", keys=["a"], values=["b"])
    rel_rev_spec = RelationSpec(name="rel_rev", keys=["b", "a"])
    query_opts = QueryOptions(order="code", limit=5, timeout=60, assert_="none")

    with CozoDB(engine="sqlite", db_path="mydb.db") as db:
        # 1. Basic constant query
        raw_result = db.script("?[] <- [['hello', 'world', 'Cozo!']]")
        print("Raw Query Result:")
        print(raw_result)

        # 2. Non-temporal relation with schema-aware multi-row put
        db.create_relation(airport_spec)
        sample_airports = [
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
        put_result = db.put_relation(
            "airport", airport_spec, sample_airports, returning=True
        )
        print("\nPut Result for 'airport':")
        print(put_result)
        result = db.query(
            select=["code", "desc", "lon", "lat"],
            from_="airport",
            where={"country": "US"},
            conditions=["lon > -0.1", "lon < 0.1"],
            options=query_opts,
        )
        print("\nNon-Temporal Query Result:")
        print(result)

        # 3. Temporal relation with multi-row put
        db.create_relation(hos_spec)
        sample_hos = [
            {"state": "US", "year": [2001, True], "hos": "Bush"},
            {"state": "US", "year": [2009, True], "hos": "Obama"},
            {"state": "US", "year": [2017, True], "hos": "Trump"},
            {"state": "US", "year": [2021, True], "hos": "Biden"},
        ]
        db.put_relation("hos", hos_spec, sample_hos, returning=True)
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

        # 4. Route relation with index
        db.create_relation(route_spec)
        sample_routes = [
            {"src": "A", "dst": "B", "weight": 1.0},
            {"src": "B", "dst": "C", "weight": 2.0},
            {"src": "C", "dst": "A", "weight": 3.0},
        ]
        db.put_relation("route", route_spec, sample_routes, returning=True)
        db.create_index("route", "dst_idx", ["dst", "src", "weight"])
        indexed_result = db.script(
            "?[src, weight] := *route:dst_idx{dst: 'B', src, weight}"
        )
        print("\nIndexed Query Result (dst='B'):")
        print(indexed_result)

        # 5. Triggers example
        db.create_relation(rel_spec)
        db.create_relation(rel_rev_spec)
        trigger_data = [{"a": 1, "b": "one"}, {"a": 2, "b": "two"}]
        db.put_relation("rel", rel_spec, trigger_data, returning=True)
        rev_result = db.script("?[b, a] := *rel_rev{b, a}")
        print("\nReverse Index Result (after triggers):")
        print(rev_result)

        # 6. Test validation with invalid data
        try:
            db.put_relation("rel", rel_spec, [{"b": "three"}])  # Missing key 'a'
        except ValueError as e:
            print("\nValidation Error (put_relation):")
            print(e)

        try:
            db.remove_rows("rel", rel_spec, [{"b": "one"}])  # Missing key 'a'
        except ValueError as e:
            print("\nValidation Error (remove_rows):")
            print(e)

        # Additional examples for other mutations
        insert_data = {
            "code": "SFO",
            "desc": "San Francisco",
            "lon": -122.375,
            "lat": 37.618,
            "country": "US",
        }
        insert_result = db.insert_relation(
            "airport", airport_spec, insert_data, returning=True
        )
        print("\nInsert Result for 'airport':")
        print(insert_result)

        remove_keys = {"code": "LAX"}
        remove_result = db.remove_rows(
            "airport", airport_spec, remove_keys, returning=True
        )
        print("\nRemove Result for 'airport':")
        print(remove_result)
