package org.hren.relation_extraction_model.data_handler

trait Idx(id: Int, e1: Int, e2: Int) {
  val this.id: Int = id
  val this.e1: Int = e1
  val this.e2: Int = e2
}
