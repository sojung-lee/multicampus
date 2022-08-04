package day11;
//도메인 객체
/**  - 해시셋은 객체를 저장하기 전에 객체의 hashCode()를 호출해서
 *    해시코드를 얻어낸다. 그리고 이미 저장되어 있는 객체들의 해시코드와 비교한다.
 *    만약 동일한 해시코드가 있다면, 다시 equals() 메서드로 두 객체를 비교해서
 *    true가 나오면 동일한 객체로 판단하고 중복저장을 하지 않는다
 * */
public class Member {
	String name;
	int age;
	
	public Member(String name, int age) {
		this.name=name;
		this.age=age;
	}
	@Override
	public int hashCode() {
		int code=name.hashCode()+age;
		return code;
	}
	@Override
	public boolean equals(Object obj) {
		if(obj instanceof Member) {
			Member user=(Member)obj;
			boolean bool=user.name.equals(this.name)&& user.age ==this.age;
			return bool;
		}else {
			return false;
		}
	}
}
