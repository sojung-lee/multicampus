package day11;
import java.util.*;
/* Set계열
 *  - 순서를 기억하지 않음
 *  - 데이터 중복을 허용하지 않는다
 *  - 구현 클래스: HashSet, TreeSet...
 *  - 해시셋은 객체를 저장하기 전에 객체의 hashCode()를 호출해서
 *    해시코드를 얻어낸다. 그리고 이미 저장되어 있는 객체들의 해시코드와 비교한다.
 *    만약 동일한 해시코드가 있다면, 다시 equals() 메서드로 두 객체를 비교해서
 *    true가 나오면 동일한 객체로 판단하고 중복저장을 하지 않는다.
 * */

public class HashSetTest {

	public static void main(String[] args) {
		
		HashSet<String> set=new HashSet<>();
		//데이터 저장
		//add(E)
		set.add("Java");
		set.add("HTML");
		set.add("CSS");
		set.add("Python");
		set.add("MySQL");
		set.add("Java");
		System.out.println("set.size(): "+set.size());
		
		//데이터 꺼내기
		//Iterator<E>	iterator()
		Iterator<String> it=set.iterator();
		while(it.hasNext()) {
			String val=it.next();
			System.out.println(val);
		}
		
		set.remove("CSS");
		System.out.println("set.size(): "+set.size());
		
		for(String str:set)
			System.out.println(str);
		
		//clear(), removeAll()
		set.clear();
		System.out.println("set.size(): "+set.size());
		
	}

}
