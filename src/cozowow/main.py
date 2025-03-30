import pandas as pd
from typing import Any, Dict, List, Union, Optional, Callable
from pycozo.client import Client, MultiTransact


def format_value(val: Any) -> str:
    """Format a Python value as a CozoScript literal."""
    if isinstance(val, str):
        return '"' + val.replace('"', '\\"') + '"'
    elif isinstance(val, bool):
        return "true" if val else "false"
    elif val is None:
        return "null"
    else:
        return str(val)


def format_validity(val: Any) -> str:
    """
    For common validity keywords, use single quotes.
    Otherwise, use format_value.
    """
    if isinstance(val, str) and val.upper() in {"NOW", "END", "ASSERT", "RETRACT"}:
        return "'" + val + "'"
    return format_value(val)


class CozoDB:
    """
    A Pythonic wrapper for CozoDB (v0.7) with time-travel support.

    Provides:
      - raw_query: for raw CozoScript queries.
      - query: a builder that constructs queries from Python parameters (supports a validity clause).
      - put, update, remove: data mutations with optional time-travel parameters.
      - Other helpers (fixed_rule, transactions, backups, etc.).
    """

    def __init__(
        self,
        engine: str = "sqlite",
        db_path: str = ":memory:",
        dataframe: bool = True,
        **options,
    ):
        self.client = Client(engine, db_path, dataframe=dataframe, **options)

    def raw_query(self, script: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if params is None:
            params = {}
        return self.client.run(script, params)

    def query(
        self,
        *,
        select: List[str],
        from_: str,
        where: Optional[Dict[str, Any]] = None,
        conditions: Optional[List[str]] = None,
        order_by: Optional[Union[str, List[str]]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        validity: Optional[Any] = None,
    ) -> Any:
        """
        Build and execute a pythonic query.

        Parameters:
          select: List of column names to return.
          from_: Stored relation name.
          where: Dict of equality conditions.
          conditions: List of additional raw conditions.
          order_by: Column(s) for ordering.
          offset: Number of rows to skip.
          limit: Maximum number of rows.
          validity: Optional time-travel clause (e.g. "NOW", "END", or a timestamp).
                    If provided, appends '@ <validity>' to the relation access.

        Returns:
          The query result.
        """
        # Rule head.
        head = "?[" + ", ".join(select) + "] :="

        # Build the first atom (relation access) with optional time travel.
        if validity is not None:
            validity_literal = format_validity(validity)
            first_atom = f"*{from_}{{{', '.join(select)}}} @ {validity_literal}"
        else:
            first_atom = f"*{from_}{{{', '.join(select)}}}"

        # Prepare additional atoms.
        atoms = [first_atom]
        if where:
            for col, value in where.items():
                atoms.append(f"{col} = {format_value(value)}")
        if conditions:
            atoms.extend(conditions)

        # Join all atoms with ", " (in one line).
        rule_body = head + " " + ", ".join(atoms)

        # Build options.
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

        # Append options on new lines.
        script = rule_body + ("\n" + "\n".join(options) if options else "")
        return self.raw_query(script)

    def put(
        self,
        relation: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        validity_field: Optional[str] = None,
        validity_value: Optional[Any] = None,
    ) -> Any:
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")
        if validity_field and validity_value is not None:
            for row in data:
                if validity_field not in row:
                    row[validity_field] = validity_value
        return self.client.put(relation, data)

    def update(
        self,
        relation: str,
        keys: Dict[str, Any],
        data: Dict[str, Any],
        validity_field: Optional[str] = None,
        validity_value: Optional[Any] = None,
    ) -> Any:
        update_data = {**keys, **data}
        if validity_field and validity_value is not None:
            if validity_field not in update_data:
                update_data[validity_field] = validity_value
        return self.client.update(relation, update_data)

    def remove(
        self,
        relation: str,
        keys: Union[Dict[str, Any], List[Dict[str, Any]]],
        validity_field: Optional[str] = None,
        retract_value: Optional[Any] = None,
    ) -> Any:
        if validity_field and retract_value is not None:
            if isinstance(keys, dict):
                keys = [keys]
            for row in keys:
                if validity_field not in row:
                    row[validity_field] = retract_value
            return self.client.put(relation, keys)
        else:
            return self.client.rm(relation, keys)

    def fixed_rule(
        self, rule_name: str, input_relations: List[str], options: Dict[str, Any]
    ) -> Any:
        options_str = ", ".join(
            f"{key}: {format_value(val)}" for key, val in options.items()
        )
        input_rel_str = ", ".join(input_relations)
        script = f"?[] <~ {rule_name}({input_rel_str}, {options_str})"
        return self.raw_query(script)

    def explain(self, script: str) -> Any:
        script = f"::explain {{ {script} }}"
        return self.raw_query(script)

    def transaction(self, read_write: bool = True) -> "CozoTransaction":
        tx = self.client.multi_transact(read_write)
        return CozoTransaction(tx)

    def export_relations(self, relations: List[str]) -> Dict[str, Any]:
        return self.client.export_relations(relations)

    def import_relations(self, relations_data: Dict[str, Any]) -> Any:
        return self.client.import_relations(relations_data)

    def backup(self, path: str) -> Any:
        return self.client.backup(path)

    def restore(self, path: str) -> Any:
        return self.client.restore(path)

    def import_from_backup(self, path: str, relations: List[str]) -> Any:
        return self.client.import_from_backup(path, relations)

    def register_callback(self, relation: str, callback: Callable) -> Any:
        return self.client.register_callback(relation, callback)

    def unregister_callback(self, callback_id: Any) -> None:
        self.client.unregister_callback(callback_id)

    def register_fixed_rule(
        self, rule_name: str, arity: int, rule_impl: Callable
    ) -> None:
        self.client.register_fixed_rule(rule_name, arity, rule_impl)

    def unregister_fixed_rule(self, rule_name: str) -> None:
        self.client.unregister_fixed_rule(rule_name)

    def close(self) -> None:
        self.client.close()


class CozoTransaction:
    def __init__(self, tx: MultiTransact):
        self.tx = tx

    def raw_query(self, script: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if params is None:
            params = {}
        return self.tx.run(script, params)

    def query(
        self,
        *,
        select: List[str],
        from_: str,
        where: Optional[Dict[str, Any]] = None,
        conditions: Optional[List[str]] = None,
        order_by: Optional[Union[str, List[str]]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        validity: Optional[Any] = None,
    ) -> Any:
        head = "?[" + ", ".join(select) + "] :="
        if validity is not None:
            validity_literal = format_validity(validity)
            first_atom = f"*{from_}{{{', '.join(select)}}} @ {validity_literal}"
        else:
            first_atom = f"*{from_}{{{', '.join(select)}}}"
        atoms = [first_atom]
        if where:
            for col, value in where.items():
                atoms.append(f"{col} = {format_value(value)}")
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
        script = rule_body + ("\n" + "\n".join(options) if options else "")
        return self.raw_query(script)

    def commit(self) -> None:
        self.tx.commit()

    def abort(self) -> None:
        self.tx.abort()

    def __enter__(self) -> "CozoTransaction":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            self.abort()
        else:
            try:
                self.commit()
            except Exception:
                self.abort()


# Example usage:
if __name__ == "__main__":
    # Use a persistent SQLite database (":memory:" is not supported by Cozo).
    db = CozoDB(engine="sqlite", db_path="mydb.db")

    # Raw query: exactly as in the tutorial.
    raw_result = db.raw_query("?[] <- [['hello', 'world', 'Cozo!']]")
    print("Raw Query Result:")
    print(raw_result)

    # Pythonic query with time travel support.
    # This should generate a query like:
    # ?[code, desc, lon, lat] := *airport{code, desc, lon, lat} @ 'NOW', country = "US", lon > -0.1, lon < 0.1
    # :order code
    # :limit 5
    pythonic_result = db.query(
        select=["code", "desc", "lon", "lat"],
        from_="airport",
        where={"country": "US"},
        conditions=["lon > -0.1", "lon < 0.1"],
        order_by="code",
        limit=5,
        validity="NOW",  # Now gets formatted as 'NOW'
    )
    print("\nPythonic Query (Time Travel) Result:")
    print(pythonic_result)

    # Data insertion with time travel: inserting a "mood" fact.
    mood_data = {"name": "me", "mood": "curious"}
    db.put("mood", mood_data, validity_field="at", validity_value="ASSERT")

    # Update example: update a user's name with a new validity timestamp.
    db.update(
        "users",
        keys={"id": 1},
        data={"name": "Alicia"},
        validity_field="at",
        validity_value="2025-01-01T00:00:00.000+00:00",
    )

    # Removal example: record a retraction instead of hard deletion.
    db.remove("users", keys={"id": 2}, validity_field="at", retract_value="RETRACT")

    # Transaction example.
    with db.transaction(read_write=True) as tx:
        tx.raw_query("?[] <- [['transaction', 'test']]")
        tx.query(select=["id", "name"], from_="users", where={"id": 1})

    db.close()
