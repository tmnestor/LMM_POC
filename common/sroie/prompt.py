"""The SROIE extraction prompt.

One definition, shared by every model under test. Comparability between
runs depends on this text being identical across them, so it lives here
rather than in a per-model notebook or config file.
"""

SROIE_PROMPT = """Read this scanned receipt and report four fields.

Write each field on its own line, in this order and this format:

company: <business name printed at the top of the receipt>
date: <date printed on the receipt>
address: <full street address of the business>
total: <final total amount paid>

Write the date exactly as the receipt prints it.
Write the total as a plain number, without a currency symbol.
Write NOT_FOUND as the value of any field the receipt does not show.
"""
