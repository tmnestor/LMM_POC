"""The SROIE extraction prompt.

One definition, shared by every model under test. Comparability between
runs depends on this text being identical across them, so it lives here
rather than in a per-model notebook or config file.

Wording history — read before "tidying" this text:

* 2026-08-14: the total line said "write the total as a plain number,
  without a currency symbol", with no example. InternVL3.5 read "plain
  number" as "no punctuation" and wrote ``490`` for 4.90 on 196 of 347
  receipts — every digit correct, the decimal point dropped — taking
  total F1 from 0.95 to 0.58. A worked example is doing real work here;
  do not remove it.
"""

SROIE_PROMPT = """Read this scanned receipt and report four fields.

Write each field on its own line, in this order and this format:

company: <business name printed at the top of the receipt>
date: <date printed on the receipt>
address: <full street address of the business>
total: <final total amount paid, e.g. 4.95>

Write the date exactly as the receipt prints it.
Write the total with its decimal point and no currency symbol, e.g. 4.95.
Write NOT_FOUND as the value of any field the receipt does not show.
"""
