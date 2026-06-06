-- Swiggy-like restaurant ordering schema for the drive-thru voice agent.
-- This schema separates restaurant catalog, branch availability, cart/order state,
-- offer handling, and retrieval documents that can be indexed into Qdrant.

CREATE TABLE restaurants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    cuisine TEXT NOT NULL,
    rating NUMERIC(2, 1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE restaurant_branches (
    id TEXT PRIMARY KEY,
    restaurant_id TEXT NOT NULL REFERENCES restaurants(id),
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    area TEXT NOT NULL,
    is_open BOOLEAN NOT NULL DEFAULT TRUE,
    fulfillment_modes TEXT NOT NULL,
    average_prep_minutes INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE menu_categories (
    id TEXT PRIMARY KEY,
    restaurant_id TEXT NOT NULL REFERENCES restaurants(id),
    name TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE menu_items (
    id TEXT PRIMARY KEY,
    restaurant_id TEXT NOT NULL REFERENCES restaurants(id),
    category_id TEXT NOT NULL REFERENCES menu_categories(id),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    is_veg BOOLEAN NOT NULL DEFAULT FALSE,
    base_price NUMERIC(10, 2) NOT NULL,
    currency CHAR(3) NOT NULL DEFAULT 'USD',
    tags TEXT NOT NULL DEFAULT '',
    synonyms TEXT NOT NULL DEFAULT ''
);

CREATE TABLE branch_menu_inventory (
    branch_id TEXT NOT NULL REFERENCES restaurant_branches(id),
    item_id TEXT NOT NULL REFERENCES menu_items(id),
    stock INTEGER NOT NULL DEFAULT 0,
    is_available BOOLEAN NOT NULL DEFAULT TRUE,
    prep_minutes INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (branch_id, item_id)
);

CREATE TABLE menu_item_variants (
    id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL REFERENCES menu_items(id),
    name TEXT NOT NULL,
    price_delta NUMERIC(10, 2) NOT NULL DEFAULT 0
);

CREATE TABLE addon_groups (
    id TEXT PRIMARY KEY,
    restaurant_id TEXT NOT NULL REFERENCES restaurants(id),
    name TEXT NOT NULL,
    min_select INTEGER NOT NULL DEFAULT 0,
    max_select INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE addons (
    id TEXT PRIMARY KEY,
    addon_group_id TEXT NOT NULL REFERENCES addon_groups(id),
    name TEXT NOT NULL,
    price NUMERIC(10, 2) NOT NULL DEFAULT 0
);

CREATE TABLE offers (
    id TEXT PRIMARY KEY,
    restaurant_id TEXT NOT NULL REFERENCES restaurants(id),
    code TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    discount_type TEXT NOT NULL,
    value NUMERIC(10, 2),
    free_item_id TEXT REFERENCES menu_items(id),
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE customers (
    id TEXT PRIMARY KEY,
    phone TEXT,
    display_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    customer_id TEXT REFERENCES customers(id),
    branch_id TEXT NOT NULL REFERENCES restaurant_branches(id),
    status TEXT NOT NULL,
    subtotal NUMERIC(10, 2) NOT NULL DEFAULT 0,
    discount_total NUMERIC(10, 2) NOT NULL DEFAULT 0,
    grand_total NUMERIC(10, 2) NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE order_items (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(id),
    item_id TEXT NOT NULL REFERENCES menu_items(id),
    variant_id TEXT REFERENCES menu_item_variants(id),
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(10, 2) NOT NULL,
    line_total NUMERIC(10, 2) NOT NULL
);

CREATE TABLE order_item_addons (
    order_item_id TEXT NOT NULL REFERENCES order_items(id),
    addon_id TEXT NOT NULL REFERENCES addons(id),
    quantity INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (order_item_id, addon_id)
);

CREATE TABLE rag_documents (
    id TEXT PRIMARY KEY,
    source_table TEXT NOT NULL,
    source_id TEXT NOT NULL,
    document_text TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    qdrant_collection TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
