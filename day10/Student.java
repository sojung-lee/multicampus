package day10;
//도메인 객체 - VO(Value Object), DTO(Data Transfer Object)
public class Student {
	
	private int id;
	private String name;
	
	public Student() {
		this(0,"아무개");
	}
	public Student(int id, String name) {
		this.id =id;
		this.name =name;
	}
	//setter, getter----
	public int getId() {
		return id;
	}
	public void setId(int id) {
		this.id = id;
	}
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}

}////////////////////////
